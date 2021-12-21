
# Samples 

## Training data
In ./ShapeNet, we provide the samples of our training data on ShapeNet. It should be mentioned that input_of_MTDCN.exr has a size of (h=256\*8, w=256\*8), where the left four columns are input partial views, and the right four columns are ground truth data. For both partial views and ground truth data, the first three columns are three channels (r,g,b) of texture maps, and the fourth column is depth map. 

## Testing data
In ./chair_1011_pix3d, we provide the one testing sample (chair_1011) from real images on Pix3D. We apply the mask provided by Pix3D to extract the object of interest. 

* *input_rgb.exr* is one test input image in .exr format.
* *object_coordinate_image.exr* is the generated object coordinate image.
* *partial_views.exr* contains 8 pairs of texture/depth views. Note that the right four columns are the copy of the left four columns (input) since there is no ground truth in testing.
* *d_completion_view_0.exr* contains the depth completions of the 8 views.
* *td_completion_view_0.exr* contains the texture/depth completions of the 8 views.