"""
Compile PentaNet_NeurIPS_Draft.md + figures into a single publication PDF.
Uses WeasyPrint for HTML→PDF rendering.
"""
import markdown
import os
import re
from weasyprint import HTML

MD_FILE = 'PentaNet_NeurIPS_Draft.md'
OUTPUT  = 'paper/PentaNet_Technical_Report.pdf'

with open(MD_FILE, 'r') as f:
    md_text = f.read()

# --- Strip LaTeX math (WeasyPrint cannot render it) ---
def delatex(text):
    """Convert LaTeX math notation to readable Unicode text."""
    # Remove display math blocks $$ ... $$ first
    def replace_display(m):
        content = m.group(1).strip()
        # Common display equations
        content = content.replace('\\max', 'max')
        content = content.replace('\\min', 'min')
        content = content.replace('\\sum', 'Σ')
        content = content.replace('\\left(', '(').replace('\\right)', ')')
        content = content.replace('\\left\\{', '{').replace('\\right\\}', '}')
        content = content.replace('\\frac{1}{d}', '(1/d)')
        content = content.replace('\\frac{1}{n}', '(1/n)')
        content = content.replace('\\text{Round}', 'Round')
        content = content.replace('\\text{Clip}', 'Clip')
        content = content.replace('\\text{detach}', 'detach')
        content = content.replace('\\bar{W}', 'W̄')
        content = content.replace('\\gamma', 'γ')
        content = content.replace('\\epsilon', 'ε')
        content = content.replace('\\mathbb{R}', 'ℝ')
        content = content.replace('\\in', '∈')
        content = content.replace('\\times', '×')
        content = content.replace('\\cdot', '·')
        content = content.replace('\\pm', '±')
        content = content.replace('\\sigma', 'σ')
        content = content.replace('\\lambda', 'λ')
        content = content.replace('\\sim', '~')
        content = content.replace('\\approx', '≈')
        content = content.replace('\\log_2(3)', 'log₂(3)')
        content = content.replace('\\log_2(5)', 'log₂(5)')
        content = content.replace('\\log_2(7)', 'log₂(7)')
        content = content.replace('\\log_2', 'log₂')
        content = content.replace('\\lceil', '⌈')
        content = content.replace('\\rceil', '⌉')
        content = content.replace('\\lfloor', '⌊')
        content = content.replace('\\rfloor', '⌋')
        content = content.replace('\\leftrightarrow', '↔')
        content = content.replace('|', '|')
        content = re.sub(r'_\{([^}]+)\}', r'_\1', content)  # subscripts
        content = re.sub(r'\^T', 'ᵀ', content)
        content = re.sub(r'\\[a-zA-Z]+', '', content)  # remove remaining commands
        content = re.sub(r'[{}]', '', content)  # remove braces
        return content

    text = re.sub(r'\$\$(.*?)\$\$', replace_display, text, flags=re.DOTALL)

    # Inline math $...$
    def replace_inline(m):
        c = m.group(1)
        c = c.replace('\\{', '{').replace('\\}', '}')
        c = c.replace('\\pm', '±')
        c = c.replace('\\bar{W}', 'W̄')
        c = c.replace('\\gamma', 'γ')
        c = c.replace('\\epsilon', 'ε')
        c = c.replace('\\sigma', 'σ')
        c = c.replace('\\lambda', 'λ')
        c = c.replace('\\sim', '~')
        c = c.replace('\\approx', '≈')
        c = c.replace('\\log_2(3)', 'log₂(3)')
        c = c.replace('\\log_2(5)', 'log₂(5)')
        c = c.replace('\\log_2(7)', 'log₂(7)')
        c = c.replace('\\log_2', 'log₂')
        c = c.replace('\\lceil', '⌈')
        c = c.replace('\\rceil', '⌉')
        c = c.replace('\\lfloor', '⌊')
        c = c.replace('\\rfloor', '⌋')
        c = c.replace('\\leftrightarrow', '↔')
        c = c.replace('\\mathbb{R}', 'ℝ')
        c = c.replace('\\text{detach}', 'detach')
        c = c.replace('\\text{Round}', 'Round')
        c = c.replace('\\text{Clip}', 'Clip')
        c = c.replace('\\times', '×')
        c = c.replace('\\cdot', '·')
        c = re.sub(r'\^T', 'ᵀ', c)
        c = re.sub(r'_\{([^}]+)\}', r'_\1', c)
        c = re.sub(r'\\[a-zA-Z]+', '', c)
        c = re.sub(r'[{}]', '', c)
        return c

    text = re.sub(r'\$([^$]+?)\$', replace_inline, text)
    return text

md_text = delatex(md_text)

# Convert markdown to HTML
# Figures are embedded directly in the markdown via ![](path) — no manual injection needed.
html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite'])

# Wrap markdown images in figure divs for PDF styling
html_body = re.sub(
    r'<img alt="(Figure \d+:[^"]*)" src="([^"]+)" />',
    r'<div class="figure"><img alt="\1" src="\2" /></div>',
    html_body
)
# Remove the italic caption lines that follow (rendered as <em> inside <p>)
# They are kept for GitHub markdown — in PDF the alt text is sufficient
html_body = re.sub(
    r'<p><em>(Figure \d+:[^<]+)</em></p>',
    r'<p class="caption">\1</p>',
    html_body
)

# Full HTML document with academic styling
full_html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Code+Pro:wght@400&display=swap');

  @page {{
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center {{ content: counter(page); font-size: 10pt; color: #666; }}
  }}

  body {{
    font-family: 'Source Serif 4', 'Georgia', serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 100%;
  }}

  h1 {{
    font-size: 20pt;
    text-align: center;
    margin-bottom: 6pt;
    line-height: 1.2;
  }}

  /* Author block */
  h1 + p, h1 + p + p, h1 + p + p + p {{
    text-align: center;
    margin: 2pt 0;
  }}

  h2 {{
    font-size: 14pt;
    margin-top: 24pt;
    margin-bottom: 8pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4pt;
  }}

  h3 {{
    font-size: 12pt;
    margin-top: 16pt;
    margin-bottom: 6pt;
  }}

  p {{
    text-align: justify;
    margin: 6pt 0;
  }}

  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12pt 0;
    font-size: 10pt;
  }}

  th, td {{
    border: 1px solid #ccc;
    padding: 6pt 10pt;
    text-align: center;
  }}

  th {{
    background-color: #f5f5f5;
    font-weight: 600;
  }}

  code {{
    font-family: 'Source Code Pro', monospace;
    font-size: 9.5pt;
    background: #f4f4f4;
    padding: 1pt 4pt;
    border-radius: 3pt;
  }}

  pre {{
    background: #f4f4f4;
    padding: 10pt;
    border-radius: 4pt;
    font-size: 9pt;
    overflow-x: auto;
  }}

  pre code {{
    background: none;
    padding: 0;
  }}

  .figure {{
    text-align: center;
    margin: 20pt 0;
    page-break-inside: avoid;
  }}

  .figure img {{
    max-width: 95%;
    height: auto;
  }}

  .caption {{
    font-size: 9.5pt;
    color: #444;
    margin-top: 6pt;
    text-align: center;
    font-style: italic;
  }}

  hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 16pt 0;
  }}

  ul, ol {{
    margin: 6pt 0;
    padding-left: 20pt;
  }}

  li {{
    margin: 3pt 0;
  }}

  strong {{
    font-weight: 600;
  }}

  /* Abstract styling */
  h2:first-of-type + p {{
    font-style: italic;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
'''

os.makedirs('paper', exist_ok=True)

print("📝 Compiling PDF...")
base_url = os.path.abspath('.') + '/'
HTML(string=full_html, base_url=base_url).write_pdf(OUTPUT)
print(f"✅ PDF written to {OUTPUT}")
