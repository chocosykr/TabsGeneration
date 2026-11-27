import subprocess
import os
import sys

def convert_html_to_pdf(html_path: str, pdf_path: str):
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)
    
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}"
    ]
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 0:
        print(f"SUCCESS: Generated PDF at {abs_pdf} ({os.path.getsize(abs_pdf):,} bytes)")
    else:
        print("ERROR generating PDF:")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(1)

if __name__ == "__main__":
    html_file = sys.argv[1] if len(sys.argv) > 1 else "docs/guitarset_comping_benchmark_report.html"
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else "docs/guitarset_comping_benchmark_report.pdf"
    convert_html_to_pdf(html_file, pdf_file)
