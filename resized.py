from PIL import Image
import os

# Lista de carpetas originales y carpetas de destino
carpetas_originales = [
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/cassava_bacterial_blight",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/cassava_brown_streak_disease",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/cassava_green_mottle",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/cassava_mosaic_disease",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/healthy",
]

# Lista de carpetas de destino
carpetas_destino = [
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/resized/cassava_bacterial_blight",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/resized/cassava_brown_streak_disease",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/resized/cassava_green_mottle",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/resized/cassava_mosaic_disease",
    "/home/usco/Downloads/cassava-leaf-disease-classification/clases/resized/healthy",
]

# Lista de extensiones de archivos de imagen que quieres procesar
extensiones_permitidas = ['.jpg', '.jpeg', '.png', '.gif']

# Tamaño al que quieres redimensionar las imágenes (ancho, alto)
nuevo_tamano = (224, 224)

# Itera sobre las carpetas originales y carpetas de destino
for carpeta_originales, carpeta_destino in zip(carpetas_originales, carpetas_destino):
    # Crea la carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Itera sobre los archivos en la carpeta original
    for archivo in os.listdir(carpeta_originales):
        # Verifica si el archivo tiene una extensión de imagen permitida
        if any(archivo.lower().endswith(ext) for ext in extensiones_permitidas):
            # Ruta completa del archivo original
            ruta_original = os.path.join(carpeta_originales, archivo)

            # Abre la imagen
            imagen = Image.open(ruta_original)

            # Redimensiona la imagen
            imagen_redimensionada = imagen.resize(nuevo_tamano)

            # Ruta completa del archivo en la carpeta de destino
            ruta_destino = os.path.join(carpeta_destino, archivo)

            # Guarda la imagen redimensionada en la carpeta de destino
            imagen_redimensionada.save(ruta_destino)

    print(f"Proceso completado para la carpeta {carpeta_originales}. Las imágenes redimensionadas se encuentran en: {carpeta_destino}")