#!/usr/bin/env python3
"""
Fixed Medical OCR Pipeline - Eliminates false positives
"""

import json
import re
import base64
import requests
import os
import torch
from typing import Dict, List, Optional, Tuple
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from dataclasses import dataclass

@dataclass
class MedicineResult:
    medicine: str
    dosage: str
    default_dosage: str
    confidence: float
    source: str
    raw_text: str = ""

class FixedMedicalOCR:
    def __init__(self, 
                 llama_model_name: str = None,
                 google_api_key: str = None,
                 openai_api_key: str = None,
                 medicine_csv_path: str = None,
                 dosage_json_path: str = None):
        """Initialize the fixed medical OCR system"""
        self.google_api_key = google_api_key
        self.openai_client = None
        self.llama_model = None
        self.llama_tokenizer = None
        
        # Setup OpenAI
        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                print("✅ OpenAI client initialized")
            except ImportError:
                print("⚠️ OpenAI not available")
        
        # Load LLaMA Vision model
        if llama_model_name:
            try:
                from unsloth import FastVisionModel
                print("🔄 Loading LLaMA Vision model...")
                self.llama_model, self.llama_tokenizer = FastVisionModel.from_pretrained(
                    llama_model_name,
                    load_in_4bit=True,
                    use_gradient_checkpointing="unsloth",
                )
                FastVisionModel.for_inference(self.llama_model)
                print("✅ LLaMA Vision model loaded!")
            except Exception as e:
                print(f"❌ Failed to load LLaMA model: {e}")
                self.llama_model = None
        
        # Load medicine database
        if medicine_csv_path and os.path.exists(medicine_csv_path):
            print("🔄 Loading medicine database...")
            self.medicine_df = pd.read_csv(medicine_csv_path)
            self._setup_strict_search()
            print(f"✅ Loaded {len(self.medicine_df)} medicines")
        else:
            print("⚠️ Medicine database not found")
            self.medicine_df = None
        
        # Load dosage database
        if dosage_json_path and os.path.exists(dosage_json_path):
            with open(dosage_json_path, 'r', encoding='utf-8') as f:
                self.dosage_db = json.load(f)
            print(f"✅ Loaded dosage database")
        else:
            self.dosage_db = {}
        
        self._setup_patterns()
        
    def _setup_strict_search(self):
        """Setup strict medicine search to prevent false positives"""
        if self.medicine_df is None:
            return
            
        try:
            # Create medicine lookup dictionaries for exact/near-exact matching
            self.exact_medicines = {}
            self.partial_medicines = {}
            
            for idx, row in self.medicine_df.iterrows():
                generic_name = str(row['generics']).strip()
                if len(generic_name) > 3 and generic_name.lower() != 'nan':
                    # Store variations for exact matching
                    variations = [
                        generic_name.lower(),
                        generic_name.lower().replace(' ', ''),
                        generic_name.lower().replace('-', ''),
                        # First word for partial matching
                        generic_name.split()[0].lower() if ' ' in generic_name else generic_name.lower()
                    ]
                    
                    medicine_info = {
                        'full_name': f"{generic_name} - {row['applicant_name']}",
                        'generic': generic_name,
                        'manufacturer': str(row['applicant_name']),
                        'form': str(row['dosage_form']),
                        'index': idx
                    }
                    
                    for variant in variations:
                        if len(variant) >= 4:  # Minimum length for medicine names
                            if variant not in self.exact_medicines:
                                self.exact_medicines[variant] = []
                            self.exact_medicines[variant].append(medicine_info)
            
            # Create word-based index for medicine names
            self.medicine_words = set()
            for medicine_info in self.exact_medicines.values():
                for med in medicine_info:
                    words = med['generic'].lower().split()
                    for word in words:
                        if len(word) >= 4:  # Only significant words
                            self.medicine_words.add(word)
            
            print(f"📊 Created strict search index with {len(self.exact_medicines)} variations")
            
        except Exception as e:
            print(f"❌ Strict search setup failed: {e}")
            self.medicine_df = None

    def _setup_patterns(self):
        """Setup patterns for dosage extraction"""
        self.dosage_patterns = [
            r'[٠-٩\d]+\.?[٠-٩\d]*\s*(?:mg|ملغ|مغ)',
            r'[٠-٩\d]+\.?[٠-٩\d]*\s*(?:قرص|أقراص|حبة|حبات)',
            r'[٠-٩\d]+\.?[٠-٩\d]*\s*(?:مل|ml)',
            r'(?:نصف|ربع|ثلث|نص)\s*(?:قرص|حبة)',
        ]
        
        self.timing_patterns = [
            r'صباحاً?|صاحاً?|صباح',
            r'مساءً?|مساً?|مساء',
            r'ظهراً?|ظهر',
            r'يومياً?|يوماً?',
            r'مرة\s*(?:واحدة|يومياً?)',
            r'مرتين',
            r'ثلاث\s*مرات',
        ]

    def extract_with_llama_vision(self, image_path: str) -> Tuple[List[str], float]:
        """Extract text using LLaMA Vision with focus on accuracy"""
        if not self.llama_model:
            return [], 0.0
            
        try:
            image = Image.open(image_path)
            
            # More focused instruction
            instruction = """
            Extract the medicine names from this prescription. Only return the actual medicine names you can clearly see.
            List each medicine name on a separate line.
            Do not include dosages, instructions, or any text that is not a medicine name.
            Be very careful to only extract what you can clearly read as medicine names.
            """
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            
            input_text = self.llama_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            
            inputs = self.llama_tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,  # Very conservative
                    do_sample=False,
                    use_cache=True,
                )
            
            generated_text = self.llama_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse the output more carefully
            medicines = self._parse_llama_output_strict(generated_text)
            return medicines, 0.9 if medicines else 0.0
            
        except Exception as e:
            print(f"❌ LLaMA Vision extraction failed: {e}")
            return [], 0.0

    def _parse_llama_output_strict(self, text: str) -> List[str]:
        """Strictly parse LLaMA output to avoid false positives"""
        medicines = []
        
        # Split by lines
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, very short lines, or obvious non-medicine text
            if (len(line) < 3 or 
                line.lower().startswith(('the', 'this', 'image', 'prescription', 'medicine names:', 'medicines:'))) :
                continue
            
            # Remove numbering and bullets
            cleaned_line = re.sub(r'^[\d\.\-\*\•\s]+', '', line)
            cleaned_line = cleaned_line.strip()
            
            # Skip if still too short
            if len(cleaned_line) < 3:
                continue
            
            # Basic validation - should contain letters
            if not re.search(r'[a-zA-Z]', cleaned_line):
                continue
            
            # Remove common prefixes/suffixes that indicate it's not a medicine name
            if any(word in cleaned_line.lower() for word in ['extract', 'from', 'prescription', 'the following']):
                continue
            
            medicines.append(cleaned_line)
        
        # Limit to reasonable number and remove duplicates
        unique_medicines = []
        seen = set()
        for med in medicines[:8]:  # Max 8 medicines per prescription
            med_lower = med.lower().strip()
            if med_lower not in seen and len(med_lower) > 2:
                seen.add(med_lower)
                unique_medicines.append(med)
        
        return unique_medicines

    def strict_medicine_search(self, query: str) -> List[Dict]:
        """Strict medicine search to prevent false positives"""
        if self.medicine_df is None or not query.strip():
            return []
        
        query_clean = query.lower().strip()
        
        # Remove dosage information from query
        query_clean = re.sub(r'\d+\.?\d*\s*(?:mg|ملغ|مغ|قرص|حبة|مل)', '', query_clean)
        query_clean = query_clean.strip()
        
        if len(query_clean) < 3:
            return []
        
        results = []
        
        # Method 1: Exact match
        if query_clean in self.exact_medicines:
            for medicine_info in self.exact_medicines[query_clean]:
                results.append({
                    'medicine': medicine_info['full_name'],
                    'similarity_score': 1.0,
                    'match_type': 'exact'
                })
        
        # Method 2: Very strict partial matching
        if not results:
            # Only match if query contains significant medicine words
            query_words = set(query_clean.split())
            significant_words = [w for w in query_words if len(w) >= 4]
            
            if significant_words:
                for word in significant_words:
                    if word in self.medicine_words:
                        # Found a real medicine word, now look for medicines containing it
                        for med_variant, medicine_list in self.exact_medicines.items():
                            if word in med_variant and len(med_variant) >= 4:
                                similarity = SequenceMatcher(None, query_clean, med_variant).ratio()
                                
                                # Very strict threshold
                                if similarity >= 0.7:
                                    for medicine_info in medicine_list:
                                        results.append({
                                            'medicine': medicine_info['full_name'],
                                            'similarity_score': similarity,
                                            'match_type': 'partial'
                                        })
        
        # Method 3: Substring matching for very similar names
        if not results:
            for med_variant, medicine_list in self.exact_medicines.items():
                # Check if query is a good substring of medicine name or vice versa
                if ((len(query_clean) >= 5 and query_clean in med_variant) or 
                    (len(med_variant) >= 5 and med_variant in query_clean)):
                    
                    similarity = SequenceMatcher(None, query_clean, med_variant).ratio()
                    if similarity >= 0.6:
                        for medicine_info in medicine_list:
                            results.append({
                                'medicine': medicine_info['full_name'],
                                'similarity_score': similarity,
                                'match_type': 'substring'
                            })
        
        # Sort by similarity and remove duplicates
        seen_medicines = set()
        unique_results = []
        
        for result in sorted(results, key=lambda x: x['similarity_score'], reverse=True):
            medicine_key = result['medicine'].lower()
            if medicine_key not in seen_medicines:
                seen_medicines.add(medicine_key)
                unique_results.append(result)
        
        return unique_results[:3]  # Return max 3 matches

    def extract_with_google_vision(self, image_path: str) -> Dict:
        """Extract text using Google Vision API"""
        if not self.google_api_key:
            return {"success": False, "error": "No Google API key"}
            
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            payload = {
                "requests": [
                    {
                        "image": {"content": image_base64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                        "imageContext": {"languageHints": ["en", "ar"]}
                    }
                ]
            }
            
            response = requests.post(
                f"https://vision.googleapis.com/v1/images:annotate?key={self.google_api_key}",
                headers={'Content-Type': 'application/json'},
                json=payload
            )
            
            if response.status_code != 200:
                return {"success": False, "error": f"API error: {response.status_code}"}
            
            result = response.json()
            if 'responses' not in result or not result['responses']:
                return {"success": False, "error": "No text detected"}
            
            resp = result['responses'][0]
            if 'fullTextAnnotation' not in resp:
                return {"success": False, "error": "No text annotation"}
            
            full_text = resp['fullTextAnnotation']['text']
            lines = full_text.split('\n') if full_text else []
            
            # Filter and process lines more carefully
            medicine_lines = []
            for line_idx, line in enumerate(lines):
                if not line.strip() or len(line.strip()) < 3:
                    continue
                
                # Skip obvious non-medicine lines
                if self._is_medicine_line(line):
                    medicine_lines.append({
                        'line_number': line_idx,
                        'text': line.strip(),
                        'dosage_info': self._extract_dosage_info(line)
                    })
            
            return {
                "success": True,
                "full_text": full_text,
                "medicine_lines": medicine_lines
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_medicine_line(self, line: str) -> bool:
        """Determine if a line likely contains a medicine name"""
        line_clean = line.strip().lower()
        
        # Skip very short lines
        if len(line_clean) < 4:
            return False
        
        # Skip lines that are clearly not medicines
        skip_patterns = [
            r'^dr\.',  # Doctor names
            r'^prof',  # Professor
            r'university',
            r'cairo',
            r'tel:',
            r'phone:',
            r'address:',
            r'التاريخ',  # Date
            r'الاسم',   # Name
            r'^\d{1,2}\/\d{1,2}\/\d{2,4}',  # Dates
            r'^\d+\s*$',  # Numbers only
            r'^[٠-٩]+\s*$',  # Arabic numbers only
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, line_clean):
                return False
        
        # Must contain some letters (English or Arabic)
        if not re.search(r'[a-zA-Zا-ي]', line):
            return False
        
        # Lines starting with Rx/ are likely medicines
        if line_clean.startswith('rx'):
            return True
        
        # Lines containing common medicine indicators
        medicine_indicators = ['mg', 'ملغ', 'قرص', 'حبة', 'مل']
        if any(indicator in line_clean for indicator in medicine_indicators):
            return True
        
        # Must have reasonable length and contain letters
        return len(line_clean) >= 4 and len(line_clean) <= 50

    def _extract_dosage_info(self, text: str) -> Dict:
        """Extract dosage information from text"""
        dosages = []
        timings = []
        
        # Extract dosages
        for pattern in self.dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dosages.extend(matches)
        
        # Extract timings
        for pattern in self.timing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timings.extend(matches)
        
        return {
            'dosages': dosages,
            'timings': timings
        }

    def get_default_dosage(self, medicine_name: str) -> str:
        """Get default dosage for a medicine"""
        if not self.dosage_db or not isinstance(self.dosage_db, list):
            return ""
        
        medicine_lower = medicine_name.lower()
        
        for medicine_entry in self.dosage_db:
            if isinstance(medicine_entry, dict) and 'enName' in medicine_entry:
                en_name = medicine_entry['enName'].lower()
                
                if (medicine_lower in en_name or en_name in medicine_lower or
                    SequenceMatcher(None, medicine_lower, en_name).ratio() > 0.7):
                    return medicine_entry.get('dosage', '')
        
        return ""

    def process_prescription(self, image_path: str) -> List[MedicineResult]:
        """Main processing pipeline with strict validation"""
        print(f"🔍 Processing prescription: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        results = []
        found_medicines = set()
        
        # Extract with LLaMA Vision
        print("🤖 Extracting medicines with LLaMA Vision...")
        llama_medicines, llama_confidence = self.extract_with_llama_vision(image_path)
        
        # Extract with Google Vision
        print("👁️ Extracting text with Google Vision...")
        google_result = self.extract_with_google_vision(image_path)
        
        # Process LLaMA results with strict validation
        if llama_medicines and llama_confidence > 0.7:
            print(f"✅ LLaMA extracted {len(llama_medicines)} potential medicines")
            
            for medicine_text in llama_medicines:
                print(f"🔎 Searching for: '{medicine_text}'")
                
                # Use strict search
                search_results = self.strict_medicine_search(medicine_text)
                
                if search_results:
                    best_match = search_results[0]
                    
                    # Only accept high-confidence matches
                    if best_match['similarity_score'] >= 0.7:
                        medicine_name = best_match['medicine']
                        medicine_key = medicine_name.lower().strip()
                        
                        if medicine_key not in found_medicines:
                            found_medicines.add(medicine_key)
                            
                            default_dosage = self.get_default_dosage(medicine_name)
                            
                            results.append(MedicineResult(
                                medicine=medicine_name,
                                dosage="غير محدد",
                                default_dosage=default_dosage or "غير متوفر",
                                confidence=best_match['similarity_score'],
                                source="llama_vision",
                                raw_text=medicine_text
                            ))
                            
                            print(f"  ✅ Added: {medicine_name} (conf: {best_match['similarity_score']:.3f})")
                    else:
                        print(f"  ❌ Rejected: low confidence ({best_match['similarity_score']:.3f})")
                else:
                    print(f"  ❌ No valid matches found")
        
        # Process Google Vision results with even stricter validation
        if google_result.get("success"):
            print("✅ Processing Google Vision results...")
            
            for line_data in google_result.get("medicine_lines", []):
                line_text = line_data["text"]
                
                # Additional filtering for Google Vision results
                if len(line_text.strip()) < 5:
                    continue
                
                # Skip lines that look like dosage instructions only
                if re.match(r'^[\d\s]+(?:قرص|حبة|مرة|يوماً|صباحاً|مساءً)', line_text.strip()):
                    continue
                
                search_results = self.strict_medicine_search(line_text)
                
                if search_results:
                    best_match = search_results[0]
                    
                    # Very strict threshold for Google Vision
                    if best_match['similarity_score'] >= 0.8:
                        medicine_name = best_match['medicine']
                        medicine_key = medicine_name.lower().strip()
                        
                        if medicine_key not in found_medicines:
                            found_medicines.add(medicine_key)
                            
                            dosage_info = line_data.get("dosage_info", {})
                            line_dosage = " ".join(dosage_info.get("dosages", []) + dosage_info.get("timings", []))
                            if not line_dosage:
                                line_dosage = "غير محدد"
                            
                            default_dosage = self.get_default_dosage(medicine_name)
                            
                            results.append(MedicineResult(
                                medicine=medicine_name,
                                dosage=line_dosage,
                                default_dosage=default_dosage or "غير متوفر",
                                confidence=best_match['similarity_score'],
                                source="google_vision",
                                raw_text=line_text
                            ))
                            
                            print(f"  ✅ Added from Google Vision: {medicine_name}")
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"🎯 Found {len(results)} validated medicine(s)")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.medicine} (conf: {result.confidence:.3f}) [from: {result.raw_text}]")
        
        return results

    def format_results(self, results: List[MedicineResult]) -> Dict:
        """Format results for output"""
        return {
            "total_medicines": len(results),
            "medicines": [
                {
                    "rank": i,
                    "medicine": result.medicine,
                    "dosage": result.dosage,
                    "default_dosage": result.default_dosage,
                    "confidence": round(result.confidence, 3),
                    "source": result.source,
                    "raw_extracted_text": result.raw_text
                }
                for i, result in enumerate(results, 1)
            ]
        }
    
    def save_simplified_json(self, results: List[MedicineResult], output_path: str = None) -> str:
        """Save simplified JSON with only medicine, dosage, and default_dosage"""
        if output_path is None:
            # Generate timestamp-based filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"prescription_medicines_{timestamp}.json"
        
        # Create simplified format
        simplified_data = []
        for i, result in enumerate(results, 1):
            simplified_data.append({
                "rank": i,
                "medicine": result.medicine,
                "dosage": result.dosage,
                "default_dosage": result.default_dosage
            })
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Simplified JSON saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Failed to save JSON: {e}")
            return ""


# Example usage
def main():
    config = {
        "llama_model_name": "elenamagdy77/Finetune_Llama_3_2_Vision_OCR",
        "google_api_key": "AIzaSyDLePMB53Q1Nud4ZG8a2XA9UUYuSLCrY6c",
        "openai_api_key": "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",
        "medicine_csv_path": "/notebooks/eda_medicines_cleaned.csv",
        "dosage_json_path": "/notebooks/Deployment/medical_products_full.json"
    }
    
    
    
    ocr_system = FixedMedicalOCR(**config)
    results = ocr_system.process_prescription("/notebooks/Deployment/pictures/7.jpg")
    
    # Display full results on screen
    formatted = ocr_system.format_results(results)
    print(json.dumps(formatted, indent=2, ensure_ascii=False))
    
    # Save simplified JSON file
    if results:
        simplified_json_path = ocr_system.save_simplified_json(results)
        print(f"\n📄 Simplified JSON file created: {simplified_json_path}")
    
if __name__ == "__main__":
    main()
