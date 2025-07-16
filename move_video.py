import os
import shutil


def move_mp4_to_parent(root_dir):
    """
    Parcourt récursivement les sous-dossiers de root_dir, trouve les fichiers MP4,
    et les déplace vers leur dossier parent principal (premier niveau sous root_dir).

    Args:
        root_dir (str): Chemin du répertoire racine contenant les dossiers principaux.
    """
    # Parcourir tous les dossiers principaux (par exemple, hello, yo)
    for main_folder in os.listdir(root_dir):
        main_folder_path = os.path.join(root_dir, main_folder)

        # Vérifier si c'est un dossier
        if not os.path.isdir(main_folder_path):
            continue

        # Parcourir récursivement les sous-dossiers
        for subdir, _, files in os.walk(main_folder_path):
            # Ignorer le dossier principal lui-même
            if subdir == main_folder_path:
                continue

            # Parcourir les fichiers dans le sous-dossier
            for file in files:
                # Vérifier si le fichier est un MP4
                if file.lower().endswith('.mp4'):
                    src_path = os.path.join(subdir, file)
                    dest_path = os.path.join(main_folder_path, file)

                    # Vérifier si un fichier avec le même nom existe déjà
                    if os.path.exists(dest_path):
                        # Ajouter un suffixe pour éviter l'écrasement
                        base, ext = os.path.splitext(file)
                        i = 1
                        while os.path.exists(os.path.join(main_folder_path, f"{base}_{i}{ext}")):
                            i += 1
                        dest_path = os.path.join(main_folder_path, f"{base}_{i}{ext}")

                    # Déplacer le fichier
                    try:
                        shutil.move(src_path, dest_path)
                        print(f"Déplacé : {src_path} -> {dest_path}")
                    except Exception as e:
                        print(f"Erreur lors du déplacement de {src_path} : {e}")


def main():
    # Chemin du répertoire racine (à modifier selon votre structure)
    root_dir = r"C:\Users\juankroos\Documents\Wondershare Filmora 9\Output"  # Exemple : "/home/user/asl_videos"

    # Vérifier si le répertoire existe
    if not os.path.exists(root_dir):
        print(f"Le répertoire {root_dir} n'existe pas.")
        return

    print(f"Parcours du répertoire : {root_dir}")
    move_mp4_to_parent(root_dir)
    print("Déplacement des fichiers MP4 terminé.")


if __name__ == "__main__":
    main()