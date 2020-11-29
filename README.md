# CUDA - Filtro Sobel

## Requisitos:
- CUDA
`nvcc` precisa estar presente no `$PATH` 
E os headers do CUDA SDK preciso estar incluso na compilação (considerou-se /usr/local/cuda/ como local padrão)

- OpenCV (Open Source Computer Vision Library) para ler e salvar arquivos jpeg, v4.2.0

No Ubuntu Linux (20.4), OpenCV pode ser instalado através do comando: `sudo apt-get install libopencv-dev`
A biblioteca e os headers são automaticamente encontrados pelo pkg-config (`pkg-config --cflags/--libs opencv4`).
Caso o pkg-config falhe, adicione manualmente os headers e as bilbiotecas do OpenCV no Makefile

## Execução
```
make 
./sobel
```

### Imagens Incluídas
- MapaMundi (https://earthobservatory.nasa.gov/blogs/elegantfigures/2011/10/06/crafting-the-blue-marble/)
 - Campbell do Andy Warhol (https://www.moma.org/learn/moma_learning/andy-warhol-campbells-soup-cans-1962/)

### Output
 O resultado será gravado na mesma pasta com o sufixo "_edge".
