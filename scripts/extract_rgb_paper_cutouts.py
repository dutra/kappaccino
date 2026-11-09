"""Export the two four-panel inset groups of a saved RGB overview as vector PDFs.

Requires PyMuPDF. Uses only the existing PDF and its JSON sidecar, preserving
the exact imaging, labels, posterior contours, and critical-curve bands.
"""
from pathlib import Path
import argparse
import hashlib
import json

import pymupdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output_directory or source.parent
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(source.with_suffix('.json').read_text())
    selections = metadata['insets']
    if any(len(selections[key]) != 4 for key in ('top', 'right')):
        raise ValueError('Expected four top and four right inset panels')
    with pymupdf.open(source) as document:
        if len(document) != 1:
            raise ValueError('Expected a single-page RGB overview')
        rectangles = [pymupdf.Rect(item['bbox']) for item in document[0].get_image_info()]
        main = max(rectangles, key=lambda rect: rect.get_area())
        insets = [rect for rect in rectangles if rect != main]
        top = sorted([rect for rect in insets if rect.y1 < main.y0], key=lambda rect: rect.x0)
        right = sorted([rect for rect in insets if rect.x0 > main.x1], key=lambda rect: rect.y0)
        if len(top) != 4 or len(right) != 4:
            raise ValueError('Could not identify the eight RGB inset images')
        for key, name, rectangles in (
            ('top', 'm0416_small_lambda_posteriors', top),
            ('right', 'm0416_fold_cutouts', right),
        ):
            side = max(rect.width for rect in rectangles)
            with pymupdf.open() as result:
                page = result.new_page(width=4*side, height=side)
                for index, clip in enumerate(rectangles):
                    target = pymupdf.Rect(index*side, 0, (index+1)*side, side)
                    page.show_pdf_page(target, document, 0, clip=clip)
                result.set_metadata({'title': name, 'subject': ', '.join(selections[key]),
                                     'creator': 'extract_rgb_paper_cutouts.py'})
                path = output / f'{name}.pdf'
                result.save(path, garbage=4, deflate=True)
            path.with_suffix('.json').write_text(json.dumps({
                'source_pdf': str(source),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'run': metadata['run'],
                'panels': selections[key],
                'source_clip_rectangles_pt': [list(rect) for rect in rectangles],
                'layout': 'one row, four panels, no added whitespace',
                'method': 'vector PDF clipping; no posterior recomputation or smoothing',
                'source_insets': selections,
            }, indent=2) + '\n')
            print(path)


if __name__ == '__main__':
    main()
