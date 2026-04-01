import json
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document

class JSONLoader:
    def __init__(self, products_path: str, warranty_path: str):
        self.products_path = Path(products_path)
        self.warranty_path = Path(warranty_path)

    def _load(self, file_path: Path) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def create_unified_doc(self) -> Document:
        products = self._load(self.products_path)
        warranties = self._load(self.warranty_path)

        warranty_map = {w['sku']: w for w in warranties}

        unified_docs = []

        for product in products:
            sku = product.get('sku')
            warranty_info = warranty_map.get(sku, {})

            # 1. Transform technical specs into a readable string
            specs_str = ", ".join([f"{k}: {v}" for k, v in product.get('specs', {}).items()])

            # 2. Synthesize the "LLM-friendly" narrative
            # This is where we bridge the two files semantically
            text_content = (
                f"Product Name: {product['name']} (SKU: {sku}).\n"
                f"Category: {product['category']}.\n"
                f"Description: {product['description']}\n"
                f"Technical Specifications: {specs_str}.\n"
                f"Warranty Details: This product has a {warranty_info.get('warranty_period', 'standard')} warranty. "
                f"Coverage: {warranty_info.get('coverage', 'N/A')}. "
                f"Exclusions: {warranty_info.get('exclusions', 'None listed')}. "
                f"Claims Process: {warranty_info.get('claims_process', 'Contact support')}."
            )
            
            # 3. Create a LangChain Document object
            # We keep the SKU and category in metadata so filters can be applied
            doc = Document(
                page_content=text_content,
                metadata={"sku": sku, "category": product.get('category', ''), "source": "product_warranty_merge"}
            )
            unified_docs.append(doc)

        return unified_docs
    
# Quick test to verify the loader works as expected
if __name__ == "__main__":
    loader = JSONLoader('data/products.json', 'data/warranty.json')
    docs = loader.create_unified_doc()

    for doc in docs:
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 80)
    