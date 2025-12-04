import os
import glob
import pandas as pd

# Caminho base
base_path = os.path.join('Kaggle', 'target_tables')

# Lista para armazenar DataFrames
dfs = []

# Percorre pastas 1_* até 8_*
for i in range(1, 9):
    tema_glob = os.path.join(base_path, f"{i}_*")
    for tema_path in glob.glob(tema_glob):
        theme = os.path.basename(tema_path).replace(f"{i}_", "")
        # Pega todos os CSVs dentro da pasta do tema
        for csv_file in glob.glob(os.path.join(tema_path, '*.csv')):
            # Subtema: nome do arquivo sem extensão
            subtheme = os.path.splitext(os.path.basename(csv_file))[0]
            # Lê o CSV
            df = pd.read_csv(csv_file)
            # Adiciona colunas de tema e subtema
            df.insert(0, 'Theme', theme)
            df.insert(1, 'Subtheme', subtheme)
            dfs.append(df)

# Concatena tudo
if dfs:
    tabela_unica = pd.concat(dfs, ignore_index=True)
    tabela_unica.to_csv('tabela_unica.csv', index=False)
    print('Arquivo tabela_unica.csv criado com sucesso!')
else:
    print('Nenhum arquivo CSV encontrado nas pastas especificadas.')
