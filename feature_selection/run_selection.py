import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../base_implementation'))
sys.path.append(os.path.dirname(__file__))

from forward_selection import ForwardSelection


def main():
    print("Iniciando Forward Selection...")
    
    chat_file = '/home/matheus/github/Clustering/StyloMetrix/Datasets/human_chat.txt'
    checkpoint_file = os.path.join(os.path.dirname(__file__), 'forward_selection_checkpoint.json')
    
    if not os.path.exists(chat_file):
        print(f"Erro: Arquivo não encontrado: {chat_file}")
        print("Por favor, ajuste o caminho no script.")
        return
    
    selector = ForwardSelection(
        chat_file_path=chat_file,
        max_features=60,
        checkpoint_file=checkpoint_file
    )
    
    results = selector.run()
    
    output_file = os.path.join(os.path.dirname(__file__), 'forward_selection_results.json')
    selector.save_results(output_file)
    
    print("\n" + "="*80)
    print("Resumo Final:")
    print("="*80)
    print(f"Features selecionadas: {results['selected_features']}")
    print(f"Número de features: {len(results['selected_features'])}")
    print(f"Acurácia final: {results['final_accuracy']:.2f}%")
    print(f"Total de iterações: {results['total_iterations']}")
    print(f"Resultados salvos em: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
