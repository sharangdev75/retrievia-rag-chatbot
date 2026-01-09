import pandas as pd
from langchain_core.documents import Document
from typing import List


def load_excel(file_path: str) -> List[Document]:
    """
    Convert Excel data into structured, queryable documents.
    Each row becomes a separate document with better context.
    """
    documents = []
    
    try:
        
        df = pd.read_excel(file_path)
        
        
        if df.columns.dtype == 'object' or all(isinstance(col, str) for col in df.columns):
            
            for idx, row in df.iterrows():
                facts = []
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        facts.append(f"{col}: {value}")
                
                if facts:
                    content = " | ".join(facts)
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_path.split("/")[-1],
                                "type": "excel",
                                "row": idx + 2  
                            }
                        )
                    )
        else:
            
            df = pd.read_excel(file_path, header=None)
            for idx, row in df.iterrows():
                cells = [str(c).strip() for c in row if pd.notna(c)]
                
                if len(cells) >= 2:
                    content = " | ".join(cells)
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_path.split("/")[-1],
                                "type": "excel",
                                "row": idx + 1
                            }
                        )
                    )
    
    except Exception as e:
        print(f"Error loading Excel file {file_path}: {str(e)}")
        return []
    
    return documents