import os
from pathlib import Path
from tika import parser


def extract_text_with_tika(file_path):
    try:
        parsed = parser.from_file(str(file_path))
        return parsed.get('content', '')
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {str(e)}")
        return None

def process_folder_simple(folder_path, output_file=None):
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Папка {folder} не существует")
        return [] 
    
    if not folder.is_dir():
        print(f"{folder} не является папкой")
        return []
    
    all_files = list(folder.iterdir())
    text_files = [f for f in all_files if f.is_file()]
    
    if not text_files:
        print(f"В папке {folder} нет файлов")
        return []
    
    results = []
    
    for file_path in text_files:
        print(f"Обработка: {file_path.name}")
        text = extract_text_with_tika(file_path)
        
        if text and text.strip():
            results.append({
                'file': file_path.name,
                'text': text.strip(),
                'path': str(file_path),
                'size': len(text.strip())
            })
            print(f"  ✓ Извлечено {len(text.strip())} символов")
        else:
            print(f"  ✗ Не удалось извлечь текст")
            results.append({
                'file': file_path.name,
                'text': '',
                'path': str(file_path),
                'size': 0,
                'error': 'Не удалось извлечь текст'
            })
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    if result['text']:
                        f.write(f"Файл: {result['file']}\n")
                        f.write("="*80 + "\n")
                        f.write(result['text'])
                        f.write("\n" + "="*80 + "\n\n")
            print(f"\nРезультаты сохранены в {output_file}")
        except Exception as e:
            print(f"Ошибка при сохранении в файл {output_file}: {str(e)}")
    
    if results:
        successful = [r for r in results if r.get('text')]
        print(f"\nОбработка завершена. Успешно: {len(successful)}/{len(results)} файлов")
        
        if not output_file:
            for result in successful[:3]:
                print(f"\nФайл: {result['file']}")
                print("="*80)
                preview = result['text'][:300].split('\n')[0] + "..." if len(result['text']) > 300 else result['text']
                print(f"Превью: {preview}")
                print("="*80)
            
            if len(successful) > 3:
                print(f"\n... и еще {len(successful) - 3} файлов")
    
    return results

# # Пример использования
# if __name__ == "__main__":
#     # Укажите путь к вашей папке
#     folder_path = "тексты/"
#     process_folder_simple(folder_path, "результаты.txt")