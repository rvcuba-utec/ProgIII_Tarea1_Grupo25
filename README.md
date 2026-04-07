# ProgIII_Tarea1_Grupo25
Programación III – Tarea 1 (Sección 14), desarrollada por el Grupo 25

## Descripción

Este proyecto implementa una biblioteca de Tensor en C++ que permite trabajar con arreglos multidimensionales (hasta 3 dimensiones) y realizar operaciones comunes como suma, resta, multiplicación, producto punto y multiplicación de matrices. También incluye transformaciones como ReLU y Sigmoid para usar en redes neuronales.

## Archivos del Proyecto

- `Tensor.h` / `Tensor.cpp` - Clase principal del Tensor con todas las operaciones
- `Transform.h` / `Transform.cpp` - Transformaciones (ReLU y Sigmoid)
- `main.cpp` - Programa de ejemplo que demuestra el uso de la biblioteca

## Cómo Compilar

Para compilar el proyecto, necesitas tener un compilador de C++ instalado (como `g++`). Ejecuta el siguiente comando en la terminal:

```bash
g++ -std=c++17 -o programa main.cpp Tensor.cpp Transform.cpp
```

Si prefieres un compilador diferente como `clang`, también funciona igual:

```bash
clang++ -std=c++17 -o programa main.cpp Tensor.cpp Transform.cpp
```

## Cómo Ejecutar

Una vez compilado, simplemente ejecuta el programa con:

```bash
./programa
```

El programa creará un tensor de entrada aleatorio de dimensiones 1000x20x20, lo pasará a través de una red neuronal simple con una capa oculta de 100 neuronas, y mostrará el resultado procesado con las activaciones ReLU y Sigmoid.

## Notas

- El proyecto usa características de C++17, así que asegúrate de compilar con el flag `-std=c++17` o superior
- Los tensores soportan broadcasting limitado (suma con dimensiones 1xN en matrices)
- La multiplicación de matrices (matmul) requiere que las dimensiones sean compatibles
