// IMPORTANT: THIS PROGRAM USES OPENCV AND CUDA!

#include <iostream>
#include "Image.h"

int main(int argc, char **argv)
{
	Image warhol("images/warhol.jpg", "Warhol", GRAYSCALE);
	Image world("images/world.jpg", "World", GRAYSCALE);

	warhol.binomial_filter(5);
	warhol.edge_detection();
	warhol.save("images/warhol_edge.jpg");

	world.binomial_filter(5);
	world.edge_detection();
	world.save("images/world_edge.jpg");

	return 0;
}
