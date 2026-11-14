#!/bin/zsh
# Compile chaque figure en deux SVG (thème clair et sombre) dans ../ : nom-light.svg, nom-dark.svg
cd "${0:A:h}"
for f in fig-*.tex; do
  n=${f%.tex}; n=${n#fig-}
  for th in light dark; do
    pre=""; [[ $th == dark ]] && pre='\def\darktheme{}'
    pdflatex -interaction=nonstopmode -jobname="$n-$th" "$pre\input{$f}" >/dev/null || { echo "ERREUR $f ($th)"; continue; }
    pdftocairo -svg "$n-$th.pdf" "../$n-$th.svg"
    rm -f "$n-$th".{aux,log,pdf}
  done
  echo "ok $n"
done
