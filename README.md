# inverse-compositional-CUDA
CUDA implementation of a multitemplate variant of the Inverse Compositional image registration algorithm (see http://wv.inf.tu-dresden.de/~wiki/Robotics/Papers/Baker/Baker-04-LK20part4.pdf)

What is multitemplate?
---------------
The original algorithm is written for one template in mind, however when we want to use the same image with different templates then we can combine these queries into one to speed up computation.

What group of transformation is used?
---------------
It is written for affine transformations, although for the first few commit (working solely on CPU) I have used a smaller set of transformations (rigid transformations) so you can figure out from that what changes are needed if you want to change thegroup to homographies.

Compilation
---------------
CUDA 9.0 and VS2017 just come out when I started developing this (school assignment). They did not work well together so I had to compile the code from command line, here is how I did it:
- compile 64bit OpenCV with the appropriate VS (I used OpenCV 3.3.0, VS2017) - the 64bit part important because CUDA only avvepts that
- create a .bat file wit the following content
```
nvcc -c -I"<OpenCV-path>\include" <src-path>\inverseCompositional.cu -o <src-path>\inverseCompositional.obj --cl-version=2017
cl /EHsc /W4 /Fe<output-name>.exe getAffine.cpp main.cpp /I "<OpenCV-path>\include" /I "<CUDA-path>\v9.0\include" /link /LIBPATH:"<CUDA-path>\v9.0\lib\x64" /LIBPATH:"<OpenCV-path>\lib\" opencv_world330.lib cuda.lib cudart_static.lib kernel32.lib inverseCompositional.obj
```
- open up the 64bit developer command line of VS (look for it in start menu)
- call .bat file
- be happy

Usage
---------------
cmd input: {path to input file}

input file layout:
{number of templates} {epsilon} {max number of iterations} {path to I image}<br></br>
{mode} {depending from mode} {path to T_1 template image}<br></br>
{mode} {depending from mode} {path to T_2 template image}<br></br>
...<br></br>
{mode} {depending from mode} {path to T_{number of templates} template image}<br></br>
<br></br>
where {mode} is in (1,2,3)<br></br>

if mode = 1:
approximate starting position of template with least squares method, based on the min(n, {number of found features}) best feature matches

1 {n} {path to T_i template image}

e.g.: 1 8 C:\Users\user\template.png
<br></br>

mode = 2:
approximate starting position of template with triangulation (cv::getAffineTransform), based on the 3 best feature matches

2 {path to T_i template image}

e.g.: 2 C:\Users\user\template.png
<br></br>

mode = 3:
direct initialization of starting position (using the upper 2x3 part of the 3x3 W warp matrix in a row-major order)

3 {a00} {a01} {tx} {a01} {a11} {ty} {path to T_i template image}

e.g.:
when W = <br></br>
0.93,  0.35, 121.2;<br></br>
-0.44,  0.87, 209.3;<br></br>
0,    0,     1<br></br>
then <br></br>
3 0.93 0.35 121.2 -0.44 0.87 209.3 C:\Users\user\template.png



Design ideas
---------------
- There is no smoothing before gradient calculation - you might want to do that yourself.
- We store everything in 1D arrays in row-major order
- The main function passes the argmuents in the above described form where the id in each dimensions struct is the position of the template in each array being passed (see code)
- We push every necessary information to GPU and only delete it at the end of the runtime but maintain only the strictly necessary part
- Output of the algorithm is stored in the same array in which we passed the input warp estimates (it is used directly)
- Apart from this in-out parameter only the dimensions array would change (order is NOT guaranteed in that container. In fact later on probably random elements will be deleted and others duplicated in it [for now only the order is changed].)
- This is because during runtime the range of examined templates are changing. For some the algorithm finishes (finds a reeeally good warp) but for others it is still running. To keep track of these and avoid unnecessary copying we use the dimensions array. In each iteration we reorder it on the CPU/host to have the active template dimensions in the first positions and then we copy this chunk to the GPU. On the GPU we only use this array to assign data to threads or blocks which then can work on the constant/precomputed data.
- Number of threads is the maximum width among the active templates. Template images and height are block dimensions. This is because (at least on my computer) the threads/block count is fairly limited.
- profit
