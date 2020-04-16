#!/usr/bin/python3

from torch import nn
import torch
from typing import Tuple


from proj2_code.torch_layer_utils import (
    get_sobel_xy_parameters,
    get_gaussian_kernel,
    ImageGradientsLayer
)

"""
Authors: Patsorn Sangkloy, Vijay Upadhya, John Lambert, Cusuh Ham,
Frank Dellaert, September 2019.
"""

class HarrisNet(nn.Module):
    """
    Implement Harris corner detector (See Szeliski 4.1.1) in pytorch by
    sequentially stacking several layers together.

    Your task is to implement the combination of pytorch module custom layers
    to perform Harris Corner detector.

    Recall that R = det(M) - alpha(trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.

    You may find the Pytorch function nn.Conv2d() helpful here.
    """

    def __init__(self):
        """
        Create a nn.Sequential() network, using 5 specific layers (not in this
        order):
          - SecondMomentMatrixLayer: Compute S_xx, S_yy and S_xy, the output is
            a tensor of size (num_image, 3, width, height)
          - ImageGradientsLayer: Compute image gradients Ix Iy. Can be
            approximated by convolving with Sobel filter.
          - NMSLayer: Perform nonmaximum suppression, the output is a tensor of
            size (num_image, 1, width, height)
          - ChannelProductLayer: Compute I_xx, I_yy and I_xy, the output is a
            tensor of size (num_image, 3, width, height)
          - CornerResponseLayer: Compute R matrix, the output is a tensor of
            size (num_image, 1, width, height)

        To help get you started, we give you the ImageGradientsLayer layer to
        compute Ix and Iy. You will need to implement all the other layers. You
        will need to combine all the layers together using nn.Sequential, where
        the output of one layer will be the input to the next layer, and so on
        (see HarrisNet diagram). You'll also need to find the right order since
        the above layer list is not sorted ;)

        Args:
        -   None

        Returns:
        -   None
        """
        super(HarrisNet, self).__init__()

        image_gradients_layer = ImageGradientsLayer() 

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        
        channel_product_layer = ChannelProductLayer()
        second_moment_matrix_layer = SecondMomentMatrixLayer()
        corner_response_layer = CornerResponseLayer()
        NMS_Layer = NMSLayer()
        self.net = nn.Sequential(image_gradients_layer, channel_product_layer, second_moment_matrix_layer,
            corner_response_layer, NMS_Layer)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of HarrisNet network. We will only test with 1
        image at a time, and the input image will have a single channel.

        Args:
        -   x: input Tensor of shape (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network,
            (num_image, 1, height, width) tensor

        """
        assert x.dim() == 4, \
            "Input should have 4 dimensions. Was {}".format(x.dim())

        return self.net(x)


class ChannelProductLayer(torch.nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy,

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing I_xx, I_yy and I_xy respectively.
    """
    def __init__(self):
        super(ChannelProductLayer, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer, which is of size
        (num_image x 2 x width x height) for Ix and Iy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for I_xx, I_yy and I_xy.

        HINT: you may find torch.cat(), torch.mul() useful here
        """

        #######################################################################
        # TODO: YOUR CODE HERE           
        #######################################################################
        #print(x.size())
        N, c, w, h = x.size()
        #[Ix,Iy] = torch.chunk(2, dim = 4)
        Ix, Iy = torch.reshape(x[:, 0, :, :], shape=(N, 1, w, h)), torch.reshape(x[:, 1, :, :], shape=(N, 1, w, h))
        #print(Ix.size())
        Ixx = torch.mul(Ix, Ix)
        Iyy = torch.mul(Iy, Iy)
        Ixy = torch.mul(Ix, Iy)
        
        output = torch.cat((Ixx, Iyy, Ixy), dim=1)
        
        #print(output.size())

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output

class SecondMomentMatrixLayer(torch.nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image I_xx, I_xy, I_yy, then
    compute S_xx, S_yy and S_xy.

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing S_xx, S_yy and S_xy, respectively

    """
    def __init__(self, ksize: torch.Tensor = 7, sigma: torch.Tensor = 5):
        """
        You may find get_gaussian_kernel() useful. You must use a Gaussian
        kernel with filter size `ksize` and standard deviation `sigma`. After
        you pass the unit tests, feel free to experiment with other values.

        Args:
        -   None

        Returns:
        -   None
        """
        super(SecondMomentMatrixLayer, self).__init__()
        self.ksize = ksize
        self.sigma = sigma

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
    
        self.kernel = get_gaussian_kernel(ksize, sigma)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer, which is of size
        (num_image, 3, width, height) for I_xx and I_yy and I_xy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for S_xx, S_yy and S_xy

        HINT:
        - You can either use your own implementation from project 1 to get the
        Gaussian kernel, OR reimplement it in get_gaussian_kernel().
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        
        output = torch.nn.functional.conv2d(x,self.kernel, padding=(self.ksize//2), groups=3)
        # add batch and channel dimensions to 2D kernel in (c,1,ksize,ksize)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class CornerResponseLayer(torch.nn.Module):
    """
    Compute R matrix.

    The output is a tensor of size (num_image, channel, height, width),
    represent corner score R

    HINT:
    - For matrix A = [a b;
                      c d],
      det(A) = ad-bc, trace(A) = a+d
    """
    def __init__(self, alpha: int=0.05):
        """
        Don't modify this __init__ function!
        """
        super(CornerResponseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        You may find torch.mul() useful here.
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        Sxx = x[0][0]
        Syy = x[0][1]
        Sxy = x[0][2]
        
        det = torch.mul(Sxx, Syy) - torch.mul(Sxy, Sxy)
        trace = Sxx + Syy
        traceSqr = torch.mul(trace, trace)
        R = det - torch.mul(self.alpha, traceSqr)

        R = R.float().reshape(1, 1, x.shape[2], x.shape[3])

        output = R
#         N, c, w, h = x.size()
#         #[Ix,Iy] = torch.chunk(2, dim = 4)
#         Sxx, Syy, Sxy = torch.reshape(x[:, 0, :, :], shape=(N, 1, w, h)), torch.reshape(x[:, 1, :, :], shape=(N, 1, w, h)), torch.reshape(x[:, 2, :, :], shape=(N, 1, w, h))
#         #return det(A) - alpha * (trace(A)^2) , A = [Sxx, Sxy Sxy, Syy] 
#         ab = torch.mul(Sxx, Syy)
#         bc = torch.mul(Sxy, Sxy)
#         detA = torch.sub(ab, bc)
#         traceA = torch.add(Sxx,Syy)
#         output = detA - 0.06*(traceA**2)
        
#         #output = torch.cat((Ixx, Iyy, Ixy), dim=1)
        

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class NMSLayer(torch.nn.Module):
    """
    NMSLayer: Perform non-maximum suppression,

    the output is a tensor of size (num_image, 1, height, width),

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d
    """
    def __init__(self):
        super(NMSLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum, and return this binary
        image, multiplied with the cornerness response values. We'll be testing
        only 1 image at a time. Input and output will be single channel images.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        (Potentially) useful functions: nn.MaxPool2d, torch.where(), torch.median()
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        zeros = torch.zeros((x.shape[2],x.shape[3]))
        
        median = torch.median(x)
        x = torch.where(x >= median, x, zeros)
        # print('x: ', x)

        self.MaxPool = nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)
        x = torch.where((x != self.MaxPool(x)), zeros, self.MaxPool(x))

        output = x


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


def get_interest_points(image: torch.Tensor, num_points: int = 4500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to return top most N x,y points with the highest confident corner
    score. Note that the return type should be a tensor. Also make sure to
    sort them in order of confidence!

    (Potentially) useful functions: torch.nonzero, torch.masked_select,
    torch.argsort

    Args:
    -   image: A tensor of shape (b,c,m,n). We will provide an image of
        (c = 1) for grayscale image.

    Returns:
    -   x: A tensor array of shape (N,) containing x-coordinates of
        interest points
    -   y: A tensor array of shape (N,) containing y-coordinates of
        interest points
    -   confidences (optional): tensor array of dim (N,) containing the
        strength of each interest point
    """

    # We initialize the Harris detector here, you'll need to implement the
    # HarrisNet() class
    harris_detector = HarrisNet()

    # The output of the detector is an R matrix of the same size as image,
    # indicating the corner score of each pixel. After non-maximum suppression,
    # most of R will be 0.
    R = harris_detector(image)

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

#     raise NotImplementedError('`get_interest_points` in `HarrisNet.py needs `
#         + 'be implemented')

    # This dummy code will compute random score for each pixel, you can
    # uncomment this and run the project notebook and see how it detects random
    # points.
    # x = torch.randint(0,image.shape[3],(num_points,))
    # y = torch.randint(0,image.shape[2],(num_points,))

    # confidences = torch.arange(num_points,0,-1)
    
    harris_detector = HarrisNet()

    R = harris_detector(image)
    
    temp = torch.ne(R, 0)

    confidences = torch.masked_select(R, temp)
    confidences = torch.sort(confidences, descending=True)[0]
    
    x = torch.nonzero(R, as_tuple=True)[3]

    y = torch.nonzero(R, as_tuple=True)[2]

    x = remove_border_vals(image, x, y ,confidences)[0]

    y = remove_border_vals(image, x, y ,confidences)[1]
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x, y, confidences



def remove_border_vals(img, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
    -   x: Torch tensor of shape (M,)
    -   y: Torch tensor of shape (M,)
    -   c: Torch tensor of shape (M,)

    Returns:
    -   x: Torch tensor of shape (N,), where N <= M (less than or equal after pruning)
    -   y: Torch tensor of shape (N,)
    -   c: Torch tensor of shape (N,)
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    xShape = x.shape[0]
    zeros = torch.zeros(xShape)
    zeros = zeros.type(torch.LongTensor)
    x = torch.where((x < (img.shape[3] - 8)) & (x > 8), x, zeros)
    mask = torch.ne(x, 0)
    x = torch.masked_select(x, mask)
    yNum = y.shape[0]
    zeros = torch.zeros(yNum)
    zeros = zeros.type(torch.LongTensor)
    y = torch.where((y < (img.shape[2] - 8)) & (y > 8), y, zeros)
    
    mask = torch.ne(y, 0)
    y = torch.masked_select(y, mask)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x, y, c
