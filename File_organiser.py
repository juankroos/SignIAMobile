import os
import shutil

# Répertoire contenant les fichiers vidéo
video_directory = "/path/to/your/videos"  # Remplacer par le chemin de votre répertoire

# Parcourir les fichiers dans le répertoire
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):  # Vérifier que c'est un fichier vidéo (ajuster l'extension si nécessaire)
        # Extraire le gloss du nom de fichier
        # Supprimer l'extension et prendre la partie avant le premier '_', ou le nom complet si pas de '_'
        base_name = os.path.splitext(filename)[0]  # Enlever l'extension (par exemple, "abdomen_00338")
        gloss = base_name.split("_")[0]  # Prendre la partie avant '_' (par exemple, "abdomen")

        # Définir le chemin du dossier de destination
        destination_dir = os.path.join(video_directory, gloss)

        # Créer le dossier s'il n'existe pas
        try:
            os.makedirs(destination_dir, exist_ok=True)
        except OSError as e:
            print(f"Erreur lors de la création du dossier {destination_dir}: {e}")
            continue

        # Chemins complets pour l'ancien et le nouveau fichier
        old_file_path = os.path.join(video_directory, filename)
        new_file_path = os.path.join(destination_dir, filename)

        # Déplacer le fichier
        try:
            shutil.move(old_file_path, new_file_path)
            print(f"Déplacé: {filename} → {new_file_path}")
        except OSError as e:
            print(f"Erreur lors du déplacement de {filename}: {e}")
    else:
        print(f"Ignorer le fichier non-vidéo: {filename}")
