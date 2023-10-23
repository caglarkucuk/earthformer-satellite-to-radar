"""Code is adapted from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir. Their license is MIT License."""

import os
from typing import List
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from earthformer.visualization.sevir.sevir_cmap import VIL_COLORS, VIL_LEVELS

VIL_LEVELS_REAL     = [0.0, 0.15, 0.26, 0.53, 0.78, 1.51, 3.53, 7.07, 12.14, 32.23, 79.26]
VIL_LEVELS_REAL_STR = ['<0.15', '0.15–0.26', '0.26–0.53', '0.53–0.78', '0.78–1.51', 
    '1.51–3.53', '3.53–7.07', '7.07–12.14', '12.14–32.23', '>32.23']  

THRESHOLDS = (0, 16, 74, 133, 160, 181, 219, 255)

def vil_cmap(encoded=True):
    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)
    # Exactly the same error occurs in the original implementation (https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/src/display/display.py).
    # ValueError: There are 10 color bins including extensions, but ncolors = 9; ncolors must equal or exceed the number of bins
    # We can not replicate the visualization in notebook (https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/notebooks/AnalyzeNowcast.ipynb) without error.
    nil = cols.pop(0)
    under = cols[0]
    # over = cols.pop()
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
    return cmap, norm

def get_cmap(type,encoded=True):
    if type.lower()=='vil':
        ### bounds = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255]
        # cmap = ListedColormap(['#4d4c4c','#28be28','#199718', '#0b690b',
        #     '#0a4a0b','#f4f400','#edac00','#f06f01','#a00101'])
        # norm=-9
        # vmin=0
        # vmax=255
        cmap, norm = vil_cmap(encoded)
        vmin, vmax = None, None
    # elif type.lower()=='ir':
    #     ## See supp4cb.xlsx to understand the decoding stuff!
    #     # Let's define a common cb for IR data!
    #     colors1 = plt.cm.plasma(np.linspace(0., 1, 128))
    #     colors2 = plt.cm.gray_r(np.linspace(0, 1, 128))
    #     # combine them and build a new colormap
    #     colors = np.vstack((colors1, colors2))
    #     cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    #     irBounds = [-7000, -5000, -3000, -1000, 1000, 3000]
    #     norm = mpl.colors.BoundaryNorm(irBounds, cmap.N, extend='both')
    #     vmin = -8315
    #     vmax = 3685
    elif type.lower()=='lght':
        cmap,norm='hot',None
        vmin,vmax=0,5
    else:
        # cmap,norm='jet',None
        # vmin,vmax=(-7000,2000) if encoded else (-70,20)
        ## See supp4cb.xlsx to understand the decoding stuff!
        # Let's define a common cb for IR data!
        colors1 = plt.cm.plasma(np.linspace(0., 1, 128))
        colors2 = plt.cm.gray_r(np.linspace(0, 1, 128))
        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        # irBounds = [-7000, -5000, -3000, -1000, 1000, 3000]
        # norm = mpl.colors.BoundaryNorm(irBounds, cmap.N, extend='both')
        norm = None
        vmin = -8315
        vmax = 2000    #    vmax = 3685
    return cmap,norm,vmin,vmax

def change_layout_np(data,
                     in_layout='NHWT', out_layout='NHWT',
                     ret_contiguous=False):
    if in_layout == out_layout: # Not sure if this should be fixed to 'NTHWC'
        pass
    else:
        # first convert to 'NHWT'
        if in_layout == 'NHWT':
            pass
        elif in_layout == 'NTHW':
            data = np.transpose(data,
                                axes=(0, 2, 3, 1))
        elif in_layout == 'NWHT':
            data = np.transpose(data,
                                axes=(0, 2, 1, 3))
        elif in_layout == 'NTCHW':
            data = data[:, :, 0, :, :]
            data = np.transpose(data,
                                axes=(0, 2, 3, 1))
        elif in_layout == 'NTHWC':
            data = data[:, :, :, :, 0]
            data = np.transpose(data,
                                axes=(0, 2, 3, 1))
        elif in_layout == 'NTWHC':
            data = data[:, :, :, :, 0]
            data = np.transpose(data,
                                axes=(0, 3, 2, 1))
        elif in_layout == 'TNHW':
            data = np.transpose(data,
                                axes=(1, 2, 3, 0))
        elif in_layout == 'TNCHW':
            data = data[:, :, 0, :, :]
            data = np.transpose(data,
                                axes=(1, 2, 3, 0))
        else:
            raise NotImplementedError

        if out_layout == 'NHWT':
            pass
        elif out_layout == 'NTHW':
            data = np.transpose(data,
                                axes=(0, 3, 1, 2))
        elif out_layout == 'NWHT':
            data = np.transpose(data,
                                axes=(0, 2, 1, 3))
        elif out_layout == 'NTCHW':
            data = np.transpose(data,
                                axes=(0, 3, 1, 2))
            data = np.expand_dims(data, axis=2)
        elif out_layout == 'NTHWC':
            data = np.transpose(data,
                                axes=(0, 3, 1, 2))
            data = np.expand_dims(data, axis=-1)
        elif out_layout == 'NTWHC':
            data = np.transpose(data,
                                axes=(0, 3, 2, 1))
            data = np.expand_dims(data, axis=-1)
        elif out_layout == 'TNHW':
            data = np.transpose(data,
                                axes=(3, 0, 1, 2))
        elif out_layout == 'TNCHW':
            data = np.transpose(data,
                                axes=(3, 0, 1, 2))
            data = np.expand_dims(data, axis=2)
        else:
            raise NotImplementedError
    if ret_contiguous:
        data = data.ascontiguousarray()
    return data

def visualize_result_kucuk(
        in_seq: np.array, target_seq: np.array,
        pred_seq_list: List[np.array],
        label_list: List[str],
        interval_real_time: float = 10.0, idx=0, norm=None, plot_stride=2,
        figsize=(20, 10*.85), 
        fs=10, # This is just the font size....
        vis_thresh=THRESHOLDS[2], 
        vis_hits_misses_fas=True):
    """
    Parameters
    ----------
    model_list: list of nn.Module
    layout_list: list of str
    in_seq:     np.array
    target_seq: np.array
    interval_real_time: float
        The minutes of each plot interval
    """
    if norm is None:
        # Values from SEVIR repo: https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/src/readers/normalizations.py
        norm = {'vil':{'scale': 47.54, 'shift': 33.44},
                'ir069':{'scale':1174.68,'shift':-3683.58},
                'ir107':{'scale':2562.43,'shift':-1552.80},
                'lght':{'scale':0.60517,'shift':0.02990}
                } 

    cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
                           'norm': get_cmap(s, encoded=True)[1],
                           'vmin': get_cmap(s, encoded=True)[2],
                           'vmax': get_cmap(s, encoded=True)[3]}
    in_len = in_seq.shape[1]
    in_channels = in_seq.shape[-1]    
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    ncols = (max_len - 1) // plot_stride + 1
    fig, ax = plt.subplots( # layout='constrained',
        nrows=in_channels+1+len(pred_seq_list),
        ncols=ncols,
        figsize=figsize)

    for i in range(0, max_len, plot_stride):
        # print(i // plot_stride, 'input')
        if i < in_len:
            xt = in_seq[idx, i, :, :, 0] * norm['ir069']['scale'] + norm['ir069']['shift'] 
            cb_069 = ax[0][i // plot_stride].imshow(xt, **cmap_dict('ir069'))

            xt = in_seq[idx, i, :, :, 1] * norm['ir107']['scale'] + norm['ir107']['shift'] 
            cb_107 = ax[1][i // plot_stride].imshow(xt, **cmap_dict('ir107'))

            xt = in_seq[idx, i, :, :, 2] * norm['lght']['scale'] + norm['lght']['shift'] 
            cb_lght = ax[2][i // plot_stride].imshow(xt, **cmap_dict('lght'))
        else: # Not sure this case is needed...
            ax[0][i // plot_stride].axis('off')
            ax[1][i // plot_stride].axis('off')
            ax[2][i // plot_stride].axis('off')

    ax[3][1].set_ylabel('Target', fontsize=fs)
    ax[4][1].set_ylabel('Prediction', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < out_len:
            xt = target_seq[idx, :, :, i] * norm['vil']['scale'] + norm['vil']['shift']
            ax[3][(i // plot_stride)+1].imshow(xt, **cmap_dict('vil'))
        else:
            ax[3][0].axis('off')

    target_seq = target_seq[idx:idx + 1] * norm['vil']['scale'] + norm['vil']['shift']
    y_preds = [pred_seq[idx:idx + 1] * norm['vil']['scale'] + norm['vil']['shift']
               for pred_seq in pred_seq_list]
    
    # Plot model predictions
    for k in range(len(pred_seq_list)):
        for i in range(0, max_len, plot_stride):
            if i < out_len:
                ax[in_channels+1 + k][(i // plot_stride)+1].imshow(y_preds[k][0, :, :, i], **cmap_dict('vil'))
            else:
                ax[in_channels+1 + k][0].axis('off')


    for i in range(0, max_len, plot_stride): # Write minutes to future!
        if i < out_len:
            ax[-1][(i // plot_stride)+1].set_title(f'{int(interval_real_time * (i + plot_stride))} Mins', 
                                                   y=-0.25, 
                                                   fontsize=14)
    for i in range(0, max_len, plot_stride): # Write minutes to past!
        ax[0][i // plot_stride].set_title(f'-{int(interval_real_time * (max_len - i - plot_stride + 2))} Mins', 
                                          fontsize=14)

    for j in range(len(ax)): # Remove ticks
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])

    #### Colorbar & Legend time!
    myticks=[-7300, -5300, -3300, -1300, 700] # 
    # Convert this to string
    myticklabels = [str(int(x/100+273)) for x in myticks]
    cb069=fig.colorbar(cb_069, ax=ax[0,:], location='left', label='C09 (Kelvin)', 
                       shrink=0.75, anchor=(-0.11, 0.3),
        ticks=myticks)
    cb069.ax.set_yticklabels(myticklabels)
    cb107=fig.colorbar(cb_107, ax=ax[1,:], location='left', label='C13 (Kelvin)', 
                       shrink=0.75, anchor=(-0.11, 0.3),
        ticks=myticks)
    cb107.ax.set_yticklabels(myticklabels)
    fig.colorbar(cb_lght, ax=ax[2,:], location='left', label='Lightning \n Counts', 
                 shrink=0.75, anchor=(-0.11, 0.3))

    # Legend of thresholds
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                             label=VIL_LEVELS_REAL_STR[i - 1])   
                       for i in range(1, num_thresh_legend + 1)]
    ax[in_channels][0].legend(handles=legend_elements, loc='center left',
                    title = 'VIL (mm)',
                    bbox_to_anchor=(-.5, -0.07), 
                    borderaxespad=0, frameon=False, 
                    )

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, ax


def save_example_vis_results(
        save_dir, save_prefix, in_seq, target_seq, pred_seq, label,
        layout='NHWT', interval_real_time: float = 10.0, idx=0,
        plot_stride=2, fs=14, norm=None):
    """
    Parameters
    ----------
    in_seq: np.array
        float value 0-1
    target_seq: np.array
        float value 0-1
    pred_seq:   np.array
        float value 0-1
    interval_real_time: float
        The minutes of each plot interval
    """
    in_seq = change_layout_np(in_seq, in_layout=layout, out_layout='NTHWC').astype(np.float32)
    target_seq = change_layout_np(target_seq, in_layout=layout).astype(np.float32)
    pred_seq = change_layout_np(pred_seq, in_layout=layout).astype(np.float32)
    fig_path = os.path.join(save_dir, f'{save_prefix}.png')
    # fig, ax = visualize_result(
    fig, ax = visualize_result_kucuk(
        in_seq=in_seq, target_seq=target_seq, pred_seq_list=[pred_seq,],
        label_list=[label, ], interval_real_time=interval_real_time, idx=idx,
        plot_stride=plot_stride, fs=fs, norm=norm)
    plt.savefig(fig_path, bbox_inches = 'tight')
    plt.close(fig)
