from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict
import re

class HTMLExtractor:
    """从 Sphinx HTML 文档中提取结构化内容"""
    
    def __init__(self):
        self.sections = []
    
    def extract(self, html_path: str) -> List[Dict]:
        """提取 HTML 中的章节内容"""
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        
        # 获取文档标题
        title = self._get_title(soup)
        
        # 提取主要内容区域
        main_content = soup.find('article') or soup.find('div', class_='content')
        if not main_content:
            return []
        
        # 按章节分割内容
        sections = []
        current_section = {
            "title": title,
            "content": [],
            "level": 1,
            "source": str(html_path)
        }
        
        for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'div']):
            if elem.name in ['h1', 'h2', 'h3', 'h4']:
                # 保存上一个章节
                if current_section["content"]:
                    sections.append(self._format_section(current_section))
                
                # 开始新章节
                current_section = {
                    "title": elem.get_text(strip=True),
                    "content": [],
                    "level": int(elem.name[1]),
                    "source": f"{html_path}#{elem.get('id', '')}",
                    "parent_title": title
                }
            
            elif elem.name == 'p':
                text = elem.get_text(strip=True)
                if text:
                    current_section["content"].append(text)
            
            elif elem.name == 'pre':
                # 代码块
                code = elem.get_text()
                current_section["content"].append(f"```\n{code}\n```")
            
            elif elem.name == 'ul':
                # 列表
                items = [li.get_text(strip=True) for li in elem.find_all('li')]
                current_section["content"].append("\n".join(f"- {item}" for item in items))
            
            elif elem.get('class') and 'admonition' in elem.get('class'):
                # 提示框（note/warning 等）
                admonition_text = elem.get_text(strip=True)
                current_section["content"].append(f"[提示] {admonition_text}")
        
        # 添加最后一个章节
        if current_section["content"]:
            sections.append(self._format_section(current_section))
        
        return sections
    
    def _get_title(self, soup: BeautifulSoup) -> str:
        """获取文档标题"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().replace(" - uCore-Tutorial-Guide-2025S 文档", "")
        h1 = soup.find('h1')
        return h1.get_text(strip=True) if h1 else "Untitled"
    
    def _format_section(self, section: Dict) -> Dict:
        """格式化章节内容"""
        full_content = "\n\n".join(section["content"])
        # 添加元信息
        header = f"【{section['title']}】"
        if section.get('parent_title') and section['parent_title'] != section['title']:
            header = f"【{section['parent_title']} > {section['title']}】"
        
        return {
            "id": f"{section['source']}::{section['title']}",
            "content": f"{header}\n\n{full_content}",
            "metadata": {
                "source": section["source"],
                "title": section["title"],
                "level": section["level"],
                "doc_title": section.get("parent_title", section["title"]),
                "char_count": len(full_content)
            }
        }

def batch_extract(html_dir: str) -> List[Dict]:
    """批量提取目录下所有 HTML 文件"""
    extractor = HTMLExtractor()
    all_sections = []
    
    for html_file in Path(html_dir).rglob("*.html"):
        try:
            sections = extractor.extract(str(html_file))
            all_sections.extend(sections)
            print(f"✓ 提取: {html_file} ({len(sections)} 个章节)")
        except Exception as e:
            print(f"✗ 失败: {html_file} - {e}")
    
    return all_sections