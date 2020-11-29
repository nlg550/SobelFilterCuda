Notas:

1. Para ler e salvar imagens (jpeg), foi utilizado a biblioteca OpenCV (Open Source Computer Vision Library) v4.2.0.
No Ubuntu Linux (20.4), essa bilbioteca pode ser instalado através do comando: sudo apt-get install libopencv-dev
A biblioteca e os headers são automaticamente encontrados pelo pkg-config (pkg-config --cflags/--libs opencv4).
Caso o pkg-config falhe, adicione os headers e as bilbiotecas do OpenCV no INCLUDES e LIBS, respectivamente

2. Foi utilizado CUDA para calcular a convolução dos filtros com as imagens, por isso, a compilação requer 
que nvcc exista no $PATH e os CUDA headers possam ser encontrados pelo compilador (/usr/local/cuda/ é o 
local padrão do CUDA) 

3. Com OpenCV e CUDA instalados, execute:
	make 
	./sobel

O programa aplicará o filtro Sobel na MapaMundi (https://earthobservatory.nasa.gov/blogs/elegantfigures/2011/10/06/crafting-the-blue-marble/) e 
na arte Campbell do Andy Warhol (https://www.moma.org/learn/moma_learning/andy-warhol-campbells-soup-cans-1962/). O resultado será gravado 
na mesma pasta com o sufixo "_edge".
