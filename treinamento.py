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
# Usada aqui para criar pastas e montar caminhos de saída
import os

# Biblioteca para operações matemáticas e geração de dados aleatórios
# Responsável por gerar dados sintéticos (uniforme/normal) e por arrays numéricos
import numpy as np

# PyTorch: biblioteca de aprendizado de máquina
# Fornece tensores, modelos, função de custo e otimizadores
import torch
import torch.nn as nn

# Biblioteca para geração de gráficos
# Usada para desenhar e salvar gráficos 2D
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# =========================================================

# Fixamos as sementes aleatórias para que os resultados
# sejam sempre os mesmos (importante para aulas e demonstrações)
# np.random.seed -> controla o gerador de números aleatórios do NumPy
# torch.manual_seed -> controla o gerador de números aleatórios do PyTorch
np.random.seed(1)
torch.manual_seed(1)

# Se True, abre as janelas dos gráficos durante a execução
# Se False, apenas salva as imagens (ideal para VS Code)
# Manter False evita travamentos em ambientes sem GUI
mostrar_janelas = False

# A cada quantas épocas salvar o gráfico do ajuste
# Ex.: 10 -> salva imagens nas épocas 10, 20, 30, ...
salvar_a_cada = 10

# Pasta principal onde todos os resultados serão salvos
# Todas as saídas (imagens) ficam dentro dessa pasta
pastas_saida = "treinamentos"

# Taxa de aprendizado (learning rate)
# Controla o "tamanho do passo" na descida da montanha
# Valores muito altos podem divergir, muito baixos podem demorar
lr = 0.001

# Número total de épocas de treinamento
# Cada época = 1 passada completa sobre os dados
num_epocas = 1000


# =========================================================
# CRIAÇÃO DAS PASTAS PARA OS RESULTADOS
# =========================================================

# Pasta para gráficos do ajuste da reta ao longo do treinamento
# Contém as imagens do ajuste a cada intervalo de épocas
os.makedirs(os.path.join(pastas_saida, "graficos_treinamento"), exist_ok=True)

# Pasta para o gráfico da função de custo
# Contém o gráfico final da evolução do erro
os.makedirs(os.path.join(pastas_saida, "grafico_custo"), exist_ok=True)



# =========================================================
# GERAÇÃO DOS DADOS (SIMULAÇÃO REALISTA)
# =========================================================

# Geração de valores aleatórios de rotação do motor (RPM)
# Intervalo típico de funcionamento de um motor
# np.random.uniform -> distribui valores uniformemente entre low e high
rotacao_rpm = np.random.uniform(low=800, high=3000, size=100)

# Geração de valores aleatórios de peso do automóvel (kg)
# Intervalo típico de veículos de passeio
peso_automovel = np.random.uniform(low=1000, high=2000, size=100)

# Parâmetros reais (desconhecidos pelo modelo)
# Eles representam o "mundo real" usado para gerar o CO₂
a1_real = 0.05   # quanto o CO₂ aumenta por RPM
a2_real = 0.02   # quanto o CO₂ aumenta por kg de peso
b_real = 30.0    # emissão base do motor em marcha lenta

# Ruído experimental:
# Simula erros de sensores, variações ambientais, etc.
# np.random.normal -> ruído Gaussiano com média 0 e desvio 10
ruido = np.random.normal(loc=0.0, scale=10.0, size=rotacao_rpm.shape)

# Emissão real de CO₂ medida (dados observados)
# Agora depende de DUAS variáveis: RPM e Peso
# Fórmula do "mundo real" com ruído
co2_medido = a1_real * rotacao_rpm + a2_real * peso_automovel + b_real + ruido

# Normalização das entradas:
# Dividimos por 1000 para evitar números muito grandes
# Isso ajuda o gradiente descendente a convergir melhor
# Mantém as features em escala próxima (1 a 3)
x1_numpy = (rotacao_rpm / 1000.0).reshape(-1, 1)      # RPM normalizado
x2_numpy = (peso_automovel / 1000.0).reshape(-1, 1)   # Peso normalizado (toneladas)
y_numpy = co2_medido.reshape(-1, 1)

# Concatenar as duas features em uma única matriz [N, 2]
# np.hstack -> concatenação horizontal de colunas
x_numpy = np.hstack([x1_numpy, x2_numpy])

# Conversão para tensores do PyTorch (float32)
# torch.from_numpy cria tensores compartilhando memória com NumPy
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# Impressões rápidas para conferência das dimensões
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
# nn.Linear cria uma camada com parâmetros treináveis (pesos + bias)
modelo = nn.Linear(in_features=2, out_features=1)


# =========================================================
# FUNÇÃO DE CUSTO E OTIMIZADOR
# =========================================================

# Função de custo: Erro Quadrático Médio (MSE)
# Mede o quão distante as previsões estão dos valores reais
# Quanto menor, melhor o ajuste
criterio = nn.MSELoss()

# Otimizador SGD (Gradiente Descendente)
# Ele usa os gradientes para atualizar m1, m2 e b
# lr define o tamanho do passo
otimizador = torch.optim.SGD(modelo.parameters(), lr=lr)


# =========================================================
# LOOP DE TREINAMENTO (GRADIENTE DESCENDENTE)
# =========================================================

# Lista para armazenar a evolução do custo
# Guardamos 1 valor por época
historico_custo = []

for epoca in range(num_epocas):

    # -------------------------
    # FORWARD PASS
    # -------------------------
    # O modelo faz uma previsão usando os valores atuais de m1, m2 e b
    # Saída tem dimensão [N, 1]
    co2_previsto = modelo(x)

    # -------------------------
    # CÁLCULO DO CUSTO
    # -------------------------
    # Mede o erro entre o CO₂ previsto e o medido
    # custo é um tensor escalar
    custo = criterio(co2_previsto, y)
    historico_custo.append(custo.item())

    # -------------------------
    # BACKWARD PASS
    # -------------------------
    # Calcula os gradientes:
    # ∂custo/∂m1, ∂custo/∂m2 e ∂custo/∂b
    # Esses gradientes ficam armazenados em modelo.weight.grad e modelo.bias.grad
    custo.backward()

    # -------------------------
    # ATUALIZAÇÃO DOS PARÂMETROS
    # -------------------------
    # Dá um passo "montanha abaixo"
    # Atualiza pesos e bias com base nos gradientes
    otimizador.step()

    # -------------------------
    # IMPRESSÕES E GRÁFICOS
    # -------------------------
    if (epoca + 1) % 10 == 0:

        # Valores atuais dos parâmetros
        # weight tem shape [1, 2]; bias é escalar
        m1_val = modelo.weight.data[0, 0].item()
        m2_val = modelo.weight.data[0, 1].item()
        b_val = modelo.bias.data.item()

        # Gradientes atuais (mostram direção e intensidade da correção)
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
    # Permite visualizar o ajuste do modelo ao longo do treinamento
    if (epoca + 1) % salvar_a_cada == 0:

        # detach() remove o tensor do grafo de gradientes
        # numpy() transforma em array para uso no matplotlib
        previsao_final = co2_previsto.detach().numpy()

        # Figura com 2 subplots: RPM vs CO2 e Peso vs CO2
        plt.figure(figsize=(14, 5))

        # Gráfico 1: RPM vs CO2 (com linha de regressão)
        plt.subplot(1, 2, 1)
        # scatter -> pontos reais medidos
        plt.scatter(rotacao_rpm, co2_medido, color='red', label='Dados reais')
        
        # Para desenhar a linha de regressão, fixamos o peso na média
        # Assim isolamos o efeito do RPM
        peso_medio = peso_automovel.mean()
        rpm_ordenado = np.sort(rotacao_rpm)
        m1_atual = modelo.weight.data[0, 0].item()
        m2_atual = modelo.weight.data[0, 1].item()
        b_atual = modelo.bias.data.item()
        # y = m1*(rpm/1000) + m2*(peso_medio/1000) + b
        linha_rpm = m1_atual * (rpm_ordenado / 1000) + m2_atual * (peso_medio / 1000) + b_atual
        # plot -> linha de regressão aprendida
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
        # Assim isolamos o efeito do peso
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
        # Salva imagem do ajuste da época atual
        plt.savefig(caminho)

        if mostrar_janelas:
            plt.show()
        plt.close()


    # Limpa os gradientes antes da próxima iteração
    # Necessário porque o PyTorch acumula gradientes por padrão
    otimizador.zero_grad()


# =========================================================
# GRÁFICO FINAL DA FUNÇÃO DE CUSTO
# Mostra a evolução do erro MSE ao longo das épocas
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
# RESULTADOS FINAIS
# Impressões para comparar parâmetros aprendidos com parâmetros reais
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
