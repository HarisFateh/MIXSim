"""Generate the two-intersection network using the exact TikZ grid styling.

Outputs next to this script:
  two_intersection_network_figure.tex  -- figure environment for an existing paper
  two_intersection_tutorial_network.pdf -- tightly cropped vector diagram
  two_intersection_tutorial_network.svg -- scalable diagram

The resulting source reproduces the O1->N1->N2->D1 horizontal path,
and O2->N1->D2, O3->N2->D3 downward vertical paths.

Requires pdflatex to render the PDF; pdftocairo to export SVG.
The .tex file is always created even if these tools are unavailable.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile

OUT = Path(__file__).resolve().parent

# Kept identical to the original 4x4 network's TikZ styles.
STYLES = r"""
\tikzset{
    >=Stealth,
    intersection/.style={
        rectangle,
        draw=teal,
        thick,
        minimum width=0.70cm,
        minimum height=0.50cm,
        inner sep=1pt,
        text=black,
        font=\footnotesize
    },
    boundary/.style={
        circle,
        draw=teal,
        thick,
        fill=teal!20,
        minimum size=3.5mm,
        inner sep=0pt
    },
    link/.style={
        ->,
        thick,
        teal
    }
}
"""

# Coordinates and labels match the desired two-intersection example.
# Important: the eastern vertical links point DOWN, from O3 to N2 to D3.
TIKZ_PICTURE = r"""\begin{tikzpicture}[every node/.append style={font=\footnotesize}]

% ------------------------------------------------------------
% Intersections
% ------------------------------------------------------------
\node[intersection] (N1) at (0,0) {$N_1$};
\node[intersection] (N2) at (3.0,0) {$N_2$};

% ------------------------------------------------------------
% Horizontal boundary nodes
% ------------------------------------------------------------
\node[boundary] (O1) at (-2.0,0) {};
\node[above=2pt of O1] {$O_1$};

\node[boundary] (D1) at (5.0,0) {};
\node[above=2pt of D1] {$D_1$};

% ------------------------------------------------------------
% Vertical boundaries at N1
% ------------------------------------------------------------
\node[boundary] (O2) at (0,1.8) {};
\node[above=2pt of O2] {$O_2$};

\node[boundary] (D2) at (0,-1.8) {};
\node[below=2pt of D2] {$D_2$};

% ------------------------------------------------------------
% Vertical boundaries at N2
% ------------------------------------------------------------
\node[boundary] (O3) at (3.0,1.8) {};
\node[above=2pt of O3] {$O_3$};

\node[boundary] (D3) at (3.0,-1.8) {};
\node[below=2pt of D3] {$D_3$};

% ------------------------------------------------------------
% Horizontal directed links
% ------------------------------------------------------------
\draw[link] (O1) -- (N1);
\draw[link] (N1) -- (N2);
\draw[link] (N2) -- (D1);

% ------------------------------------------------------------
% Vertical directed links: origins above, destinations below
% ------------------------------------------------------------
\draw[link] (O2) -- (N1);
\draw[link] (N1) -- (D2);
\draw[link] (O3) -- (N2);
\draw[link] (N2) -- (D3);

\end{tikzpicture}"""

FIGURE_CODE = (
    "% ============================================================\n"
    "% (b) TWO-INTERSECTION NETWORK\n"
    "% ============================================================\n"
    "\\begin{figure}[htbp]\n"
    "\\centering\n"
    "\\begin{minipage}[c]{0.60\\textwidth}\n"
    "\\centering\n"
    "\\begin{adjustbox}{max width=\\linewidth}\n"
    + TIKZ_PICTURE + "\n"
    "\\end{adjustbox}\n"
    "\\end{minipage}\n"
    "\\caption{Two-intersection network.}"
    "\\label{fig:two_intersection_network}\n"
    "\\end{figure}\n"
)

# For rendering, standalone produces a cropped vector diagram without margins.
STANDALONE_SOURCE = (
    "\\documentclass[tikz,border=7pt]{standalone}\n"
    "\\usepackage{tikz}\n"
    "\\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric}\n"
    + STYLES + "\n"
    "\\begin{document}\n"
    + TIKZ_PICTURE + "\n"
    "\\end{document}\n"
)


def main():
    tex_path = OUT / 'two_intersection_network_figure.tex'
    tex_path.write_text(FIGURE_CODE, encoding='utf-8')
    print(f"Wrote {tex_path}")

    if shutil.which('pdflatex') is None:
        print('pdflatex unavailable; .tex figure is ready for Overleaf.')
        return

    with tempfile.TemporaryDirectory(prefix='two_intersection_') as tmp:
        tmp = Path(tmp)
        standalone_tex = tmp / 'network_standalone.tex'
        standalone_tex.write_text(STANDALONE_SOURCE, encoding='utf-8')
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             '-output-directory', str(tmp), str(standalone_tex)],
            cwd=tmp, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        pdf_out = OUT / 'two_intersection_tutorial_network.pdf'
        shutil.copy2(tmp / 'network_standalone.pdf', pdf_out)
        print(f"Wrote {pdf_out}")

    if shutil.which('pdftocairo'):
        svg_out = OUT / 'two_intersection_tutorial_network.svg'
        subprocess.run(['pdftocairo', '-svg', str(pdf_out), str(svg_out)], check=True)
        print(f"Wrote {svg_out}")


if __name__ == '__main__':
    main()
