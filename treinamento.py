# -*- coding: utf-8 -*-
"""
Gradiente Descendente - Exemplo Motor + Peso x Emissão de CO2

Este código demonstra, passo a passo, o funcionamento do
método do Gradiente Descendente aplicado a um modelo linear múltiplo.

Contexto físico:
- Entrada (x1): rotação do motor (RPM)
- Entrada (x2): peso do automóvel (kg)
- Saída (y): emissão de CO₂ (g/s)

Objetivo:
Encontrar os melhores valores de m1, m2 e b no modelo:
    y = m1*x1 + m2*x2 + b
que minimizam o erro entre o valor previsto e o valor medido.
"""

# =========================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =========================================================

# Biblioteca para lidar com diretórios e arquivos do sistema
import os

# Biblioteca para operações matemáticas e geração de dados aleatórios
import numpy as np

# PyTorch: biblioteca de aprendizado de máquina
import torch
import torch.nn as nn

# Biblioteca para geração de gráficos
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# =========================================================

# Fixamos as sementes aleatórias para que os resultados
# sejam sempre os mesmos (importante para aulas e demonstrações)
np.random.seed(1)
torch.manual_seed(1)

# Se True, abre as janelas dos gráficos durante a execução
# Se False, apenas salva as imagens (ideal para VS Code)
mostrar_janelas = False

# A cada quantas épocas salvar o gráfico do ajuste
salvar_a_cada = 10

# Pasta principal onde todos os resultados serão salvos
pastas_saida = "regressao_motor_peso_co2"

# Taxa de aprendizado (learning rate)
# Controla o "tamanho do passo" na descida da montanha
lr = 0.001

# Número total de épocas de treinamento
num_epocas = 1000


# =========================================================
# CRIAÇÃO DAS PASTAS PARA OS RESULTADOS
# =========================================================

# Pasta para gráficos do ajuste da reta ao longo do treinamento
os.makedirs(os.path.join(pastas_saida, "graficos_treinamento"), exist_ok=True)

# Pasta para o gráfico da função de custo
os.makedirs(os.path.join(pastas_saida, "grafico_custo"), exist_ok=True)

# Pasta para o gráfico 3D
os.makedirs(os.path.join(pastas_saida, "grafico_3d"), exist_ok=True)


# =========================================================
# GERAÇÃO DOS DADOS (SIMULAÇÃO REALISTA)
# =========================================================

# Geração de valores aleatórios de rotação do motor (RPM)
# Intervalo típico de funcionamento de um motor
rotacao_rpm = np.random.uniform(low=800, high=3000, size=100)

# Geração de valores aleatórios de peso do automóvel (kg)
# Intervalo típico de veículos de passeio
peso_automovel = np.random.uniform(low=1000, high=2000, size=100)

# Parâmetros reais (desconhecidos pelo modelo)
# Eles representam o "mundo real"
a1_real = 0.05   # quanto o CO₂ aumenta por RPM
a2_real = 0.02   # quanto o CO₂ aumenta por kg de peso
b_real = 30.0    # emissão base do motor em marcha lenta

# Ruído experimental:
# Simula erros de sensores, variações ambientais, etc.
ruido = np.random.normal(loc=0.0, scale=10.0, size=rotacao_rpm.shape)

# Emissão real de CO₂ medida (dados observados)
# Agora depende de DUAS variáveis: RPM e Peso
co2_medido = a1_real * rotacao_rpm + a2_real * peso_automovel + b_real + ruido

# Normalização das entradas:
# Dividimos por 1000 para evitar números muito grandes
# Isso ajuda o gradiente descendente a convergir melhor
x1_numpy = (rotacao_rpm / 1000.0).reshape(-1, 1)      # RPM normalizado
x2_numpy = (peso_automovel / 1000.0).reshape(-1, 1)   # Peso normalizado (toneladas)
y_numpy = co2_medido.reshape(-1, 1)

# Concatenar as duas features em uma única matriz [N, 2]
x_numpy = np.hstack([x1_numpy, x2_numpy])

# Conversão para tensores do PyTorch (float32)
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

print("Shapes dos dados:", x.shape, y.shape)
print("Features: [RPM/1000, Peso/1000]")


# =========================================================
# DEFINIÇÃO DO MODELO MATEMÁTICO
# =========================================================

# Modelo linear múltiplo:
# y = m1*x1 + m2*x2 + b
# onde:
# - m1 é o peso para RPM (inclinação em relação ao RPM)
# - m2 é o peso para Peso do automóvel
# - b é o bias (intercepto)
modelo = nn.Linear(in_features=2, out_features=1)


# =========================================================
# FUNÇÃO DE CUSTO E OTIMIZADOR
# =========================================================

# Função de custo: Erro Quadrático Médio (MSE)
# Mede o quão distante as previsões estão dos valores reais
criterio = nn.MSELoss()

# Otimizador SGD (Gradiente Descendente)
# Ele usa os gradientes para atualizar m1, m2 e b
otimizador = torch.optim.SGD(modelo.parameters(), lr=lr)


# =========================================================
# LOOP DE TREINAMENTO (GRADIENTE DESCENDENTE)
# =========================================================

# Lista para armazenar a evolução do custo
historico_custo = []

for epoca in range(num_epocas):

    # -------------------------
    # FORWARD PASS
    # -------------------------
    # O modelo faz uma previsão usando os valores atuais de m1, m2 e b
    co2_previsto = modelo(x)

    # -------------------------
    # CÁLCULO DO CUSTO
    # -------------------------
    # Mede o erro entre o CO₂ previsto e o medido
    custo = criterio(co2_previsto, y)
    historico_custo.append(custo.item())

    # -------------------------
    # BACKWARD PASS
    # -------------------------
    # Calcula os gradientes:
    # ∂custo/∂m1, ∂custo/∂m2 e ∂custo/∂b
    custo.backward()

    # -------------------------
    # ATUALIZAÇÃO DOS PARÂMETROS
    # -------------------------
    # Dá um passo "montanha abaixo"
    otimizador.step()

    # -------------------------
    # IMPRESSÕES E GRÁFICOS
    # -------------------------
    if (epoca + 1) % 10 == 0:

        # Valores atuais dos parâmetros
        m1_val = modelo.weight.data[0, 0].item()
        m2_val = modelo.weight.data[0, 1].item()
        b_val = modelo.bias.data.item()

        grad_m1 = modelo.weight.grad[0, 0].item()
        grad_m2 = modelo.weight.grad[0, 1].item()
        grad_b = modelo.bias.grad.item()

        print('Epoch:', epoca + 1)
        print('Custo:', f"{custo.item():.20f}")
        print('Coeficientes:')
        print(f"m1 (RPM)  = {m1_val:.20f} | gradiente m1 = {grad_m1:.20f}")
        print(f"m2 (Peso) = {m2_val:.20f} | gradiente m2 = {grad_m2:.20f}")
        print(f"b (bias)  = {b_val:.20f} | gradiente b  = {grad_b:.20f}")
        print("-" * 50)

    # Gráfico 2D a cada intervalo (mostra cada variável separadamente)
    if (epoca + 1) % salvar_a_cada == 0:

        previsao_final = co2_previsto.detach().numpy()

        # Figura com 2 subplots: RPM vs CO2 e Peso vs CO2
        plt.figure(figsize=(14, 5))

        # Gráfico 1: RPM vs CO2 (com linha de regressão)
        plt.subplot(1, 2, 1)
        plt.scatter(rotacao_rpm, co2_medido, color='red', label='Dados reais')
        
        # Para desenhar a linha de regressão, fixamos o peso na média
        peso_medio = peso_automovel.mean()
        rpm_ordenado = np.sort(rotacao_rpm)
        m1_atual = modelo.weight.data[0, 0].item()
        m2_atual = modelo.weight.data[0, 1].item()
        b_atual = modelo.bias.data.item()
        # y = m1*(rpm/1000) + m2*(peso_medio/1000) + b
        linha_rpm = m1_atual * (rpm_ordenado / 1000) + m2_atual * (peso_medio / 1000) + b_atual
        plt.plot(rpm_ordenado, linha_rpm, 'b-', linewidth=2, label='Modelo aprendido')
        
        plt.xlabel("Rotação do motor (RPM)")
        plt.ylabel("Emissão de CO₂ (g/s)")
        plt.title(f"RPM vs CO₂ | Época {epoca+1} (peso fixo = {peso_medio:.0f} kg)")
        plt.legend()
        plt.grid(True)

        # Gráfico 2: Peso vs CO2 (com linha de regressão)
        plt.subplot(1, 2, 2)
        plt.scatter(peso_automovel, co2_medido, color='red', label='Dados reais')
        
        # Para desenhar a linha de regressão, fixamos o RPM na média
        rpm_medio = rotacao_rpm.mean()
        peso_ordenado = np.sort(peso_automovel)
        # y = m1*(rpm_medio/1000) + m2*(peso/1000) + b
        linha_peso = m1_atual * (rpm_medio / 1000) + m2_atual * (peso_ordenado / 1000) + b_atual
        plt.plot(peso_ordenado, linha_peso, 'b-', linewidth=2, label='Modelo aprendido')
        
        plt.xlabel("Peso do automóvel (kg)")
        plt.ylabel("Emissão de CO₂ (g/s)")
        plt.title(f"Peso vs CO₂ | Época {epoca+1} (RPM fixo = {rpm_medio:.0f})")
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Custo = {custo.item():.4f}", fontsize=12)
        plt.tight_layout()

        caminho = os.path.join(
            pastas_saida,
            "graficos_treinamento",
            f"epoch_{epoca+1}.png"
        )
        plt.savefig(caminho)

        if mostrar_janelas:
            plt.show()
        plt.close()

    # Limpa os gradientes antes da próxima iteração
    otimizador.zero_grad()


# =========================================================
# GRÁFICO FINAL DA FUNÇÃO DE CUSTO
# =========================================================

plt.figure(figsize=(7, 5))
plt.plot(historico_custo, 'b-')
plt.xlabel("Épocas")
plt.ylabel("Erro (MSE)")
plt.title("Evolução da Função de Custo")
plt.grid(True)

custo_path = os.path.join(
    pastas_saida,
    "grafico_custo",
    "custo.png"
)
plt.savefig(custo_path)

if mostrar_janelas:
    plt.show()
plt.close()


# =========================================================
# GRÁFICO 3D DO PLANO DE REGRESSÃO
# =========================================================

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dados reais
ax.scatter(rotacao_rpm, peso_automovel, co2_medido,
           c='red', marker='o', alpha=0.6, label='Dados reais')

# Criar grade para o plano de regressão
rpm_grid = np.linspace(rotacao_rpm.min(), rotacao_rpm.max(), 20)
peso_grid = np.linspace(peso_automovel.min(), peso_automovel.max(), 20)
RPM_mesh, PESO_mesh = np.meshgrid(rpm_grid, peso_grid)

# Calcular previsões para a grade (usando os coeficientes aprendidos)
m1_final = modelo.weight.data[0, 0].item()
m2_final = modelo.weight.data[0, 1].item()
b_final = modelo.bias.data.item()

# Lembre-se que o modelo usa valores normalizados (divididos por 1000)
CO2_mesh = m1_final * (RPM_mesh / 1000) + m2_final * (PESO_mesh / 1000) + b_final

# Plano de regressão
ax.plot_surface(RPM_mesh, PESO_mesh, CO2_mesh, alpha=0.5, color='blue')

ax.set_xlabel('Rotação do Motor (RPM)')
ax.set_ylabel('Peso do Automóvel (kg)')
ax.set_zlabel('Emissão de CO₂ (g/s)')
ax.set_title('Plano de Regressão Linear Múltipla')
ax.legend()

grafico_3d_path = os.path.join(
    pastas_saida,
    "grafico_3d",
    "plano_regressao.png"
)
plt.savefig(grafico_3d_path, dpi=150)

if mostrar_janelas:
    plt.show()
plt.close()


# =========================================================
# RESULTADOS FINAIS
# =========================================================

print("\nTreinamento finalizado.")
print("Parâmetros aprendidos pelo modelo:")
print(f"m1 (inclinação RPM)  = {modelo.weight.data[0, 0].item():.6f}")
print(f"m2 (inclinação Peso) = {modelo.weight.data[0, 1].item():.6f}")
print(f"b  (intercepto)      = {modelo.bias.data.item():.6f}")

print("\nParâmetros reais usados na simulação:")
print(f"a1 (RPM)  = {a1_real:.6f}")
print(f"a2 (Peso) = {a2_real:.6f}")
print(f"b (bias)  = {b_real:.6f}")

# Comparação dos coeficientes (considerando a normalização)
# O modelo aprendeu m1 e m2 para entradas normalizadas (x/1000)
# Para comparar com a1_real e a2_real, precisamos dividir por 1000
print("\nComparação (coeficientes ajustados para escala original):")
print(f"m1/1000 = {modelo.weight.data[0, 0].item()/1000:.6f} vs a1_real = {a1_real:.6f}")
print(f"m2/1000 = {modelo.weight.data[0, 1].item()/1000:.6f} vs a2_real = {a2_real:.6f}")
