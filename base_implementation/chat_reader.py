def read_human_chat(file_path):
    chat_data = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('Human'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        chat_entry = {
                            'nomePessoa': parts[0].strip(),
                            'texto': parts[1].strip()
                        }
                        chat_data.append(chat_entry)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        
    return chat_data
