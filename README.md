# Proyecto de Transfer Learning

Este proyecto utiliza Python y varias bibliotecas científicas para tareas de machine learning. A continuación se presentan las instrucciones necesarias para crear un entorno Conda y configurar todas las dependencias usando un archivo `enviroment.yml`.

---

## Crear un entorno Conda desde cero

Si deseas crear un entorno manualmente, puedes hacerlo con los siguientes pasos:

1. Crear el entorno llamado `aves` con una versión específica de Python:

```bash
conda create --name aves python=3.10
```

Una vez finalizada la creación, activar el entorno para empezar a trabajar dentro de él. Esto asegura que todos los paquetes que instales y ejecutes estén contenidos dentro del entorno aves.

```bash
conda activate aves
```

A partir de este punto, cualquier instalación o ejecución se realizará dentro del entorno aves, sin afectar otras configuraciones de Python en tu sistema.

## Crear un entorno desde el archivo enviroment.yml

1. Para configurar el entorno automáticamente con todas las dependencias del proyecto:

2. Asegúrate de estar en el directorio raíz del proyecto (donde se encuentra el archivo `enviroment.yml`).

Ejecuta el siguiente comando para crear el entorno:

```bash
conda env create -f enviroment.yml
```

3. Una vez creado, activa el entorno:

```bash
conda activate aves
```
El entorno creado se llamará `aves`, según lo definido en el archivo `enviroment.yml`.

## Actualizar el entorno si cambias el archivo `enviroment.yml`

```bash
conda env update -f enviroment.yml --prune
```
El argumento --prune eliminará los paquetes que ya no estén definidos en el archivo.
