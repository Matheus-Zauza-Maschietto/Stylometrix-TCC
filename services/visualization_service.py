import matplotlib.pyplot as plt
import numpy as np

class VisualizationService:
    @staticmethod
    def create_accuracy_bar_chart(correct_predictions: int, incorrect_predictions: int, output_path: str = 'accuracy_chart.png'):
        total = correct_predictions + incorrect_predictions
        
        if total == 0:
            print("Erro: Nenhuma predição foi feita.")
            return
        
        categories = ['Acertos', 'Falhas']
        values = [correct_predictions, incorrect_predictions]
        colors = ['#2ecc71', '#e74c3c']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            percentage = (value / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Quantidade de Predições', fontsize=12, fontweight='bold')
        ax.set_title('Taxa de Acerto vs Falha na Identificação de Autor', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, max(values) * 1.2)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        accuracy = (correct_predictions / total) * 100
        plt.figtext(0.5, 0.02, f'Acurácia Total: {accuracy:.2f}% | Total de Predições: {total}',
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
        
        plt.show()
        
    @staticmethod
    def create_detailed_bar_chart(correct_predictions: int, incorrect_predictions: int, 
                                  human1_correct: int = 0, human1_incorrect: int = 0,
                                  human2_correct: int = 0, human2_incorrect: int = 0,
                                  output_path: str = 'detailed_accuracy_chart.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        categories_total = ['Acertos', 'Falhas']
        values_total = [correct_predictions, incorrect_predictions]
        colors_total = ['#2ecc71', '#e74c3c']
        
        bars1 = ax1.bar(categories_total, values_total, color=colors_total, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        
        total = correct_predictions + incorrect_predictions
        for bar, value in zip(bars1, values_total):
            height = bar.get_height()
            percentage = (value / total) * 100 if total > 0 else 0
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('Quantidade', fontsize=11, fontweight='bold')
        ax1.set_title('Resultado Geral', fontsize=12, fontweight='bold')
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)
        ax1.set_ylim(0, max(values_total) * 1.2 if max(values_total) > 0 else 1)
        
        if human1_correct + human1_incorrect > 0 and human2_correct + human2_incorrect > 0:
            x = np.arange(2)
            width = 0.35
            
            bars2 = ax2.bar(x - width/2, [human1_correct, human2_correct], width, 
                          label='Acertos', color='#2ecc71', alpha=0.8, edgecolor='black')
            bars3 = ax2.bar(x + width/2, [human1_incorrect, human2_incorrect], width, 
                          label='Falhas', color='#e74c3c', alpha=0.8, edgecolor='black')
            
            for bars in [bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax2.set_ylabel('Quantidade', fontsize=11, fontweight='bold')
            ax2.set_title('Resultado por Autor', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Human 1', 'Human 2'])
            ax2.legend()
            ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax2.set_axisbelow(True)
        else:
            ax2.text(0.5, 0.5, 'Dados por autor não disponíveis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Resultado por Autor', fontsize=12, fontweight='bold')
        
        accuracy = (correct_predictions / total) * 100 if total > 0 else 0
        plt.suptitle(f'Análise de Identificação de Autor - Acurácia: {accuracy:.2f}%', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico detalhado salvo em: {output_path}")
        plt.show()

    @staticmethod
    def create_confusion_matrix(results: dict = None, output_path: str = 'confusion_matrix.png'):
        if results is None:
            print("Erro: 'results' não fornecido. Não é possível gerar a matriz de confusão.")
            return None

        h1_correct = results.get('human1_correct', 0)
        h1_incorrect = results.get('human1_incorrect', 0)
        h2_incorrect = results.get('human2_incorrect', 0)
        h2_correct = results.get('human2_correct', 0)

        cm = np.array([[h1_correct, h1_incorrect],
                       [h2_incorrect, h2_correct]])

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        thresh = cm.max() / 2. if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}",
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=14, fontweight='bold')

        ax.set_xlabel('Predito', fontsize=11, fontweight='bold')
        ax.set_ylabel('Real', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Human 1', 'Human 2'])
        ax.set_yticklabels(['Human 1', 'Human 2'])
        ax.set_title('Matriz de Confusão (2x2)', fontsize=13, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Contagem', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        return cm
