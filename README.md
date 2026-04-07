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
g++ -std=c++17 -o tensor main.cpp Tensor.cpp Transform.cpp
```

Y otra opción, usando CMakeLists.txt

```bash
mkdir build && cd build
cmake ..
make
```

## Cómo Ejecutar

Una vez compilado, simplemente se ejecuta con:

```bash
./tensor
```
