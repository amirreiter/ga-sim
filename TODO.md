1. Create combined CPU kernel:

Copy specular kernel

Instead of doing specular and diffuse in 2 passes:

Create a random switch that uses specular/diffuse ratio as the selector
for what the next ray's type is.

Do not multiply energy by the specular/diffuse ratio, the random selection
accounts for this.

This prevents the specular/diffuse paths creating an O(n^2) tree, keeping each
path linear.

2. Create GPGPU version:

Wgpu now includes hardware RT ray queries
Wisc needs to be updated to allow this.
