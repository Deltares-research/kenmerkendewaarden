# kenmerkendewaarden
Voor het afleiden van kengetallen als slotgemiddelden, gemiddelde getijkrommen, havengetallen, overschrijdingsfrequenties op basis van waterstandsmetingen

## installation
- open anaconda prompt
- `conda create --name kw_env python=3.11 git -y`
- `conda activate kw_env`
- `pip install git+https://github.com/Deltares-research/kenmerkendewaarden`

## contributing
- open git bash window
- `git clone https://github.com/deltares-research/kenmerkendewaarden`
- `cd kenmerkendewaarden`
- open anaconda prompt
- `conda create --name kw_env python=3.11 git spyder -c conda-forge -y`
- `conda activate kw_env`
- `pip install -e .[dev,examples]`
- more contributing guidelines available on dfm_tools repos: https://deltares.github.io/dfm_tools/CONTRIBUTING.html
