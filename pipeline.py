#!/usr/bin/env python3
"""
Complete Integrated Medical OCR Pipeline
Combines LLaMA Vision, Google Vision API, and OpenAI for comprehensive prescription processing
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
from difflib import get_close_matches, SequenceMatcher
from dataclasses import dataclass

# Import handling for different environments
try:
    from unsloth import FastVisionModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("⚠️ Unsloth not available - LLaMA Vision path will be disabled")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ OpenAI not available - text correction will use rule-based only")

@dataclass
class MedicineResult:
    """Data class to hold final medicine extraction results"""
    medicine: str
    dosage: str
    default_dosage: str
    confidence: float
    source: str

class IntegratedMedicalOCR:
    def __init__(self, 
                 llama_model_name: str = None,
                 google_api_key: str = None,
                 openai_api_key: str = None,
                 medicine_csv_path: str = None,
                 dosage_json_path: str = None):
        """Initialize the integrated medical OCR system"""
        self.google_api_key = google_api_key
        self.openai_client = None
        self.llama_model = None
        self.llama_tokenizer = None
        
        # Setup OpenAI if available
        if openai_api_key and HAS_OPENAI:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            print("✅ OpenAI client initialized")
        else:
            print("⚠️ OpenAI not available - using rule-based correction only")
        
        # Load LLaMA Vision model if available
        if llama_model_name and HAS_UNSLOTH:
            try:
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
        else:
            print("⚠️ LLaMA Vision not available")
        
        # Load medicine database for fuzzy search
        if medicine_csv_path and os.path.exists(medicine_csv_path):
            print("🔄 Loading medicine database...")
            self.medicine_df = pd.read_csv(medicine_csv_path)
            self._setup_fuzzy_search()
            print(f"✅ Loaded {len(self.medicine_df)} medicines for fuzzy search")
        else:
            print("⚠️ Medicine database not found - fuzzy search disabled")
            self.medicine_df = None
        
        # Load dosage database
        if dosage_json_path and os.path.exists(dosage_json_path):
            print("🔄 Loading dosage database...")
            with open(dosage_json_path, 'r', encoding='utf-8') as f:
                self.dosage_db = json.load(f)
            
            # Debug: Check the structure of dosage database
            if isinstance(self.dosage_db, list):
                print(f"✅ Loaded {len(self.dosage_db)} medicine entries (list format)")
                if self.dosage_db:
                    sample_entry = self.dosage_db[0]
                    print(f"📋 Sample entry keys: {list(sample_entry.keys())}")
                    if 'enName' in sample_entry:
                        print(f"📋 Sample medicine: {sample_entry['enName']}")
            elif isinstance(self.dosage_db, dict):
                print(f"✅ Loaded {len(self.dosage_db)} default dosages (dict format)")
        else:
            print("⚠️ Dosage database not found - default dosages disabled")
            self.dosage_db = {}
        
        # Setup patterns for dosage extraction
        self._setup_dosage_patterns()
        
    def _setup_fuzzy_search(self):
        """Setup TF-IDF vectorizer for fuzzy medicine search"""
        if self.medicine_df is None:
            return
            
        try:
            # Preprocess medicine data
            self.medicine_df['combined_text'] = (
                self.medicine_df['generics'].astype(str) + ' ' + 
                self.medicine_df['applicant_name'].astype(str) + ' ' + 
                self.medicine_df['dosage_form'].astype(str)
            ).str.lower()
            
            # Remove special characters
            self.medicine_df['combined_text'] = self.medicine_df['combined_text'].apply(
                lambda x: re.sub(r'[^\w\s]', ' ', str(x))
            )
            
            # Create TF-IDF vectors
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.vectors = self.vectorizer.fit_transform(self.medicine_df['combined_text'])
            
            # Extract medicine terms for typo correction
            self.medicine_terms = set()
            for generic in self.medicine_df['generics']:
                if pd.notna(generic):
                    terms = str(generic).lower().split()
                    self.medicine_terms.update(terms)
                    
        except Exception as e:
            print(f"❌ Fuzzy search setup failed: {e}")
            self.medicine_df = None
    
    def _setup_dosage_patterns(self):
        """Setup regex patterns for dosage extraction"""
        self.dosage_patterns = [
            r'[٠-٩\d]+\s*(?:وحد[هة]|حب[هة]|قرص|أقراص|مل|ملغ|غرام)',
            r'(?:نصف|ربع|ثلث)\s*(?:وحد[هة]|حب[هة]|قرص)',
        ]
        
        self.timing_patterns = [
            r'صباحاً?|صاحاً?',
            r'مساءً?|مساً?',
            r'ظهراً?',
            r'ليلاً?',
            r'يومياً?|يوماً?',
            r'مرة\s*(?:واحدة|يومياً?)',
            r'مرتين',
            r'ثلاث\s*مرات',
        ]
        
        self.meal_patterns = [
            r'قبل\s*الأكل',
            r'بعد\s*الأكل',
            r'مع\s*الأكل',
            r'بالأكل',
            r'على\s*الريق',
        ]

    def extract_with_llama_vision(self, image_path: str) -> Tuple[str, float]:
        """Extract medicine names using fine-tuned LLaMA Vision model"""
        if not self.llama_model:
            return "", 0.0
            
        try:
            image = Image.open(image_path)
            
            instruction = """
            You are an expert in Optical Character Recognition (OCR). Your task is to accurately extract and transcribe ALL medicine names from prescription images. List each medicine name on a separate line. Focus only on the medicine names, not dosages or instructions. Extract all visible medicine names from the prescription.
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
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                    use_cache=True,
                )
            
            generated_text = self.llama_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean up the output and handle multiple medicines
            lines = generated_text.split('\n')
            medicine_names = []
            
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith('the image') and len(line) > 2:
                    line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
                    line = re.sub(r'\s+', ' ', line)
                    if len(line) > 2:
                        medicine_names.append(line)
            
            combined_medicines = ' '.join(medicine_names) if medicine_names else generated_text
            return combined_medicines, 0.9
            
        except Exception as e:
            print(f"❌ LLaMA Vision extraction failed: {e}")
            return "", 0.0

    def extract_with_google_vision(self, image_path: str) -> Dict:
        """Extract Arabic text using Google Vision API"""
        if not self.google_api_key:
            return {"success": False, "error": "No Google API key provided"}
            
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            payload = {
                "requests": [
                    {
                        "image": {"content": image_base64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                        "imageContext": {"languageHints": ["ar"]}
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
            
            processed_lines = []
            all_dosages = []
            all_timings = []
            all_meal_instructions = []
            
            for line in lines:
                if not line.strip() or not self._is_arabic_text(line):
                    continue
                
                corrected_line = self._correct_arabic_text(line)
                dosage_info = self._extract_dosage_info(corrected_line)
                
                processed_lines.append({
                    'original': line,
                    'corrected': corrected_line,
                    'dosage_info': dosage_info
                })
                
                all_dosages.extend(dosage_info.get('dosages', []))
                all_timings.extend(dosage_info.get('timings', []))
                all_meal_instructions.extend(dosage_info.get('meal_instructions', []))
            
            return {
                "success": True,
                "full_text": full_text,
                "processed_lines": processed_lines,
                "dosages": all_dosages,
                "timings": all_timings,
                "meal_instructions": all_meal_instructions
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        if not text:
            return False
        
        arabic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                    arabic_chars += 1
        
        return total_chars > 0 and (arabic_chars / total_chars) > 0.3

    def _correct_arabic_text(self, text: str) -> str:
        """Correct Arabic text using OpenAI with fallback to rule-based"""
        if not self.openai_client:
            return self._rule_based_correction(text)
            
        try:
            prompt = f"""صحح هذا النص الطبي العربي فقط إذا كان يحتوي على أخطاء إملائية واضحة:

"{text}"

إذا كان النص صحيحاً، أعده كما هو. إذا كان خطأ واضح، صححه فقط.

النص المصحح:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "أنت مصحح نصوص طبية. صحح الأخطاء الواضحة فقط."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            corrected = response.choices[0].message.content.strip()
            corrected = corrected.strip('"').strip("'").strip()
            
            if len(corrected) > len(text) * 2:
                return self._rule_based_correction(text)
            
            return corrected
            
        except Exception as e:
            print(f"⚠️ OpenAI correction failed, using rule-based: {e}")
            return self._rule_based_correction(text)

    def _rule_based_correction(self, text: str) -> str:
        """Simple rule-based corrections for common OCR errors"""
        corrections = {
            'صاحا': 'صباحا', 'صاحاً': 'صباحاً',
            'مسا': 'مساء', 'مساً': 'مساءً',
            'يوما': 'يوماً', 'وحده': 'وحدة',
            'حبه': 'حبة', 'هيام': 'أيام',
            'عرف': 'قرص', 'مساءاره': 'مساء',
            'ب الأكل': 'بالأكل',
            '۲': '٢', '۰': '٠', '۱': '١', '۳': '٣',
            '۴': '٤', '۵': '٥', '۶': '٦', '۷': '٧',
            '۸': '٨', '۹': '٩',
        }
        
        corrected = text
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        return corrected

    def _extract_dosage_info(self, text: str) -> Dict:
        """Extract dosage information from Arabic text"""
        result = {
            'dosages': [],
            'timings': [],
            'meal_instructions': []
        }
        
        for pattern in self.dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = self._arabic_to_english_numbers(match)
                result['dosages'].append({
                    'original': match,
                    'normalized': normalized
                })
        
        for pattern in self.timing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result['timings'].extend(matches)
        
        for pattern in self.meal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result['meal_instructions'].extend(matches)
        
        return result

    def _arabic_to_english_numbers(self, text: str) -> str:
        """Convert Arabic numerals to English"""
        mapping = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        for arabic, english in mapping.items():
            text = text.replace(arabic, english)
        return text

    def fuzzy_search_medicine(self, query: str, top_n: int = 5) -> List[Dict]:
        """Search for medicines using fuzzy matching"""
        if self.medicine_df is None:
            return []
            
        try:
            corrected_query = self._correct_medicine_typos(query)
            query_vector = self.vectorizer.transform([corrected_query.lower()])
            similarity_scores = cosine_similarity(query_vector, self.vectors).flatten()
            
            top_indices = similarity_scores.argsort()[-top_n:][::-1]
            top_scores = similarity_scores[top_indices]
            
            results = []
            for idx, score in zip(top_indices, top_scores):
                if score > 0.01:
                    results.append({
                        'medicine': str(self.medicine_df.iloc[idx]['generics']),
                        'manufacturer': str(self.medicine_df.iloc[idx]['applicant_name']),
                        'dosage_form': str(self.medicine_df.iloc[idx]['dosage_form']),
                        'similarity_score': float(score)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Fuzzy search failed: {e}")
            return []

    def _correct_medicine_typos(self, query: str) -> str:
        """Correct potential typos in medicine names"""
        if not hasattr(self, 'medicine_terms'):
            return query
            
        words = query.lower().split()
        corrected_words = []

        for word in words:
            if len(word) <= 3:
                corrected_words.append(word)
                continue

            matches = get_close_matches(word, self.medicine_terms, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def get_default_dosage(self, medicine_name: str) -> str:
        """Get default dosage for a medicine from the dosage database"""
        if not self.dosage_db:
            return ""
        
        # Handle case where dosage_db is a list of objects (your actual format)
        if isinstance(self.dosage_db, list):
            medicine_lower = medicine_name.lower()
            
            # Direct lookup by enName
            for medicine_entry in self.dosage_db:
                if isinstance(medicine_entry, dict) and 'enName' in medicine_entry:
                    en_name = medicine_entry['enName'].lower()
                    
                    # Exact match
                    if medicine_lower == en_name:
                        return medicine_entry.get('dosage', '')
                    
                    # Partial match
                    if medicine_lower in en_name or en_name in medicine_lower:
                        return medicine_entry.get('dosage', '')
                    
                    # Word-based matching
                    medicine_words = set(medicine_lower.split())
                    en_name_words = set(en_name.split())
                    
                    if medicine_words and en_name_words:
                        common_words = medicine_words.intersection(en_name_words)
                        if len(common_words) / len(medicine_words) > 0.5:
                            return medicine_entry.get('dosage', '')
            
            # Fuzzy matching for list format
            best_match = None
            best_score = 0
            
            for medicine_entry in self.dosage_db:
                if isinstance(medicine_entry, dict) and 'enName' in medicine_entry:
                    en_name = medicine_entry['enName']
                    similarity = SequenceMatcher(None, medicine_lower, en_name.lower()).ratio()
                    
                    if similarity > best_score and similarity > 0.6:
                        best_score = similarity
                        best_match = medicine_entry
            
            if best_match:
                return best_match.get('dosage', '')
        
        # Handle case where dosage_db is a dictionary (fallback)
        elif isinstance(self.dosage_db, dict):
            if medicine_name in self.dosage_db:
                return self.dosage_db[medicine_name]
            
            medicine_lower = medicine_name.lower()
            for med_name, dosage in self.dosage_db.items():
                if medicine_lower in med_name.lower() or med_name.lower() in medicine_lower:
                    return dosage
            
            for med_name, dosage in self.dosage_db.items():
                similarity = SequenceMatcher(None, medicine_lower, med_name.lower()).ratio()
                if similarity > 0.8:
                    return dosage
        
        return ""

    def _extract_all_dosages_in_order(self, google_result: Dict) -> List[str]:
        """Extract all dosages in the order they appear in the text"""
        if not google_result.get("success"):
            return []
        
        dosages_in_order = []
        
        for line_data in google_result.get("processed_lines", []):
            line_dosage_info = line_data.get("dosage_info", {})
            line_dosages = []
            
            # Add specific dosages from this line
            for dosage_info in line_dosage_info.get("dosages", []):
                line_dosages.append(dosage_info["normalized"])
            
            # Add timing and meal instructions from this line
            line_dosages.extend(line_dosage_info.get("timings", []))
            line_dosages.extend(line_dosage_info.get("meal_instructions", []))
            
            if line_dosages:
                combined_dosage = " ".join(line_dosages)
                dosages_in_order.append(combined_dosage)
                print(f"  📋 Extracted dosage: {combined_dosage}")
        
        return dosages_in_order
    
    def _split_medicine_names(self, llama_medicine: str) -> List[str]:
        """Split extracted medicine text into individual medicine names"""
        potential_medicines = []
        
        # Method 1: Split by spaces
        words = llama_medicine.split()
        potential_medicines.extend(words)
        
        # Method 2: Split by common delimiters
        delimited_split = re.split(r'[,;\n\|]+', llama_medicine)
        potential_medicines.extend([m.strip() for m in delimited_split if m.strip()])
        
        # Method 3: Split by lines
        if '\n' in llama_medicine:
            lines = llama_medicine.split('\n')
            potential_medicines.extend([line.strip() for line in lines if line.strip()])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_medicines = []
        for med in potential_medicines:
            med_clean = med.strip()
            if med_clean and len(med_clean) > 2 and med_clean.lower() not in seen:
                seen.add(med_clean.lower())
                unique_medicines.append(med_clean)
        
        return unique_medicines
    
    def _get_base_medicine_name(self, medicine_name: str) -> str:
        """Extract base medicine name without concentration or form"""
        base_name = medicine_name
        
        # Remove concentration patterns
        base_name = re.sub(r'\s*\d+\.?\d*\s*[Mm][Gg]\b', '', base_name)
        base_name = re.sub(r'\s*\d+\.?\d*\s*ملغ\b', '', base_name)
        
        # Remove form indicators
        forms_to_remove = ['Penfill', 'Flexpen', 'Protect', 'Plus', 'SR', 'XR', 'Tablet', 'Capsule']
        for form in forms_to_remove:
            base_name = re.sub(rf'\s*{form}\b', '', base_name, flags=re.IGNORECASE)
        
        return base_name.strip()
    
    def _find_existing_medicine_variant(self, results: List[MedicineResult], base_medicine: str) -> Optional[MedicineResult]:
        """Find if a medicine variant already exists in results"""
        for result in results:
            result_base = self._get_base_medicine_name(result.medicine)
            if base_medicine.lower() == result_base.lower():
                return result
        return None

    def process_prescription(self, image_path: str) -> List[MedicineResult]:
        """Main processing pipeline that integrates both paths"""
        print(f"🔍 Processing prescription: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        results = []
        found_medicines = set()
        
        # Path 1: LLaMA Vision for medicine extraction
        print("🤖 Extracting medicines with LLaMA Vision...")
        llama_medicine, llama_confidence = self.extract_with_llama_vision(image_path)
        
        # Path 2: Google Vision for Arabic text and dosage
        print("👁️ Extracting Arabic text with Google Vision...")
        google_result = self.extract_with_google_vision(image_path)
        
        # Extract all dosages in order
        all_dosages = self._extract_all_dosages_in_order(google_result)
        print(f"📏 Found {len(all_dosages)} dosages in order")
        
        # Process LLaMA Vision results
        if llama_medicine and llama_confidence >= 0.3:
            print(f"✅ LLaMA extracted: {llama_medicine}")
            
            # Split into individual medicine names
            potential_medicines = self._split_medicine_names(llama_medicine)
            print(f"🔍 Split into {len(potential_medicines)} potential medicines: {potential_medicines}")
            
            dosage_index = 0
            
            for potential_med in potential_medicines:
                if len(potential_med.strip()) < 3:
                    continue
                    
                print(f"🔎 Searching for: '{potential_med}'")
                fuzzy_results = self.fuzzy_search_medicine(potential_med, top_n=3)
                
                if fuzzy_results:
                    best_match = fuzzy_results[0]
                    if best_match['similarity_score'] > 0.15:
                        medicine_name = best_match['medicine']
                        medicine_key = medicine_name.lower().strip()
                        
                        # Check if this is a variant of an already found medicine
                        base_medicine = self._get_base_medicine_name(medicine_name)
                        existing_medicine = self._find_existing_medicine_variant(results, base_medicine)
                        
                        if existing_medicine:
                            # This is a variant - use same dosage
                            print(f"  🔄 Variant of {existing_medicine.medicine}: {medicine_name}")
                            assigned_dosage = existing_medicine.dosage
                        else:
                            # New medicine - assign next dosage
                            if dosage_index < len(all_dosages):
                                assigned_dosage = all_dosages[dosage_index]
                                dosage_index += 1
                                print(f"  📏 Assigned dosage #{dosage_index}: {assigned_dosage}")
                            else:
                                assigned_dosage = "غير محدد"
                                print(f"  ⚠️ No more dosages available")
                        
                        if medicine_key not in found_medicines:
                            found_medicines.add(medicine_key)
                            
                            default_dosage = self.get_default_dosage(medicine_name)
                            
                            results.append(MedicineResult(
                                medicine=medicine_name,
                                dosage=assigned_dosage,
                                default_dosage=default_dosage or "غير متوفر",
                                confidence=best_match['similarity_score'],
                                source="llama_vision"
                            ))
                            
                            print(f"  ✅ Added: {medicine_name} (conf: {best_match['similarity_score']:.3f})")
        
        # Process Google Vision results for additional medicines
        if google_result.get("success"):
            print("✅ Google Vision text processed")
            
            for line_idx, line_data in enumerate(google_result.get("processed_lines", [])):
                corrected_text = line_data["corrected"]
                
                if len(corrected_text.strip()) < 3:
                    continue
                
                fuzzy_results = self.fuzzy_search_medicine(corrected_text, top_n=1)
                
                for fuzzy_match in fuzzy_results:
                    if fuzzy_match['similarity_score'] >= 0.25:
                        medicine_name = fuzzy_match['medicine']
                        medicine_key = medicine_name.lower().strip()
                        
                        if medicine_key not in found_medicines:
                            found_medicines.add(medicine_key)
                            
                            # Extract dosage from this specific line
                            line_dosage_info = line_data.get("dosage_info", {})
                            line_dosages = []
                            
                            for dosage_info in line_dosage_info.get("dosages", []):
                                line_dosages.append(dosage_info["normalized"])
                            
                            line_dosages.extend(line_dosage_info.get("timings", []))
                            line_dosages.extend(line_dosage_info.get("meal_instructions", []))
                            
                            line_dosage = " ".join(line_dosages) if line_dosages else "غير محدد"
                            default_dosage = self.get_default_dosage(medicine_name)
                            
                            results.append(MedicineResult(
                                medicine=medicine_name,
                                dosage=line_dosage,
                                default_dosage=default_dosage or "غير متوفر",
                                confidence=fuzzy_match['similarity_score'],
                                source=f"google_vision_line_{line_idx}"
                            ))
                            
                            print(f"  ✅ Added from Google Vision: {medicine_name}")
        
        # Sort results by confidence while maintaining groups
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"🎯 Found {len(results)} medicine(s)")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.medicine} (dosage: {result.dosage})")
        
        return results

    def format_results(self, results: List[MedicineResult]) -> Dict:
        """Format results for final output"""
        formatted = {
            "total_medicines": len(results),
            "medicines": []
        }
        
        for i, result in enumerate(results, 1):
            formatted["medicines"].append({
                "rank": i,
                "medicine": result.medicine,
                "dosage": result.dosage,
                "default_dosage": result.default_dosage,
                "confidence": round(result.confidence, 3),
                "source": result.source
            })
        
        return formatted


def main():
    """Main function to test the integrated system"""
    
    config = {
        "llama_model_name": "elenamagdy77/Finetune_Llama_3_2_Vision_OCR",
        "google_api_key": "AIzaSyDLePMB53Q1Nud4ZG8a2XA9UUYuSLCrY6c",
        "openai_api_key": "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",
        "medicine_csv_path": "/notebooks/eda_medicines_cleaned.csv",
        "dosage_json_path": "/notebooks/Deployment/medical_products_full.json"
    }
    
    test_images = ["/notebooks/Deployment/10.jpg"]
    
    print("🚀 INTEGRATED MEDICAL OCR PIPELINE")
    print("="*60)
    
    try:
        ocr_system = IntegratedMedicalOCR(**config)
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
            
        print(f"\n{'='*80}")
        print(f"🧪 PROCESSING: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        try:
            results = ocr_system.process_prescription(image_path)
            
            if not results:
                print("❌ No medicines detected")
                continue
            
            formatted_results = ocr_system.format_results(results)
            
            print("\n🎯 FINAL INTEGRATED OUTPUT:")
            print("="*60)
            
            for medicine_data in formatted_results["medicines"]:
                print(f"\n🏥 Medicine #{medicine_data['rank']}:")
                print(f"  💊 Medicine: {medicine_data['medicine']}")
                print(f"  📏 Dosage: {medicine_data['dosage']}")
                print(f"  🎯 Default Dosage: {medicine_data['default_dosage']}")
                print(f"  📊 Confidence: {medicine_data['confidence']}")
                print(f"  🔍 Source: {medicine_data['source']}")
            
            print(f"\n✅ Successfully processed {len(results)} medicine(s)")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()