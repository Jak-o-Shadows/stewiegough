# sample_three.py

"""

Link : https://white-wheels.hatenadiary.org/entry/20100327/p5

"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib  # TODO: Remove
matplotlib.interactive(True)
#matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar2Wx
from mpl_toolkits.mplot3d import Axes3D

import wx


class MyCanvasPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        #------------

        self.parent = parent
    
        self.figure = Figure()
        self.figure.set_facecolor((1.,1.,1.))
        self.axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvas(self, -1, self.figure)
        # self.canvas.SetBackgroundColour(wx.Colour(100, 100, 100))

        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()

        size = tuple(self.parent.GetClientSize())
        self.canvas.SetSize(740, 740)
        self.figure.set_size_inches(float(size[0])/self.figure.get_dpi(),
                                    float(size[1])/self.figure.get_dpi())

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.RIGHT| wx.TOP | wx.GROW)

        self.SetSizer(sizer)
        self.Fit()

    def plot(self):
        x = np.arange(-3, 3, 0.25)
        y = np.arange(-3, 3, 0.25)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X)+ np.cos(Y)

        ax = Axes3D(self.figure)
        ax.plot_wireframe(X, Y, Z)


class SliderButtonPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.parent = parent

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        k = wx.Slider(self)

        
        
        
        sizer.Add(k)


        self.SetSizer(sizer)
        self.Fit()




class IntSliderTextCtrl(wx.Panel):
    def __init__(self, parent, lower, upper, start, label=None):
        wx.Panel.__init__(self, parent)

        self.parent = parent
        vbox = wx.BoxSizer(wx.HORIZONTAL)

        if label: label = wx.StaticText(self, label=label)

        self.slider = wx.Slider(self, -1, start, lower, upper, style=wx.SL_HORIZONTAL)
        self.textControl = wx.TextCtrl(self, -1, value="", style=wx.TE_CENTER|wx.TE_PROCESS_ENTER)

        self.slider.Bind(wx.EVT_SLIDER, lambda event: self._sliderUpdate(event, self.slider,self.textControl))
        self.textControl.Bind(wx.EVT_TEXT_ENTER, lambda event: self._tcUpdate(event, self.slider,self.textControl))

        if label: vbox.Add(label, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        vbox.Add(self.slider, 1, wx.EXPAND | wx.TOP | wx.RIGHT | wx.LEFT | wx.BOTTOM)
        vbox.Add(self.textControl, 1, wx.TOP | wx.BOTTOM)

        self.SetValue(start)


        self.SetSizer(vbox)
        self.Fit()

    def _sliderUpdate(self, event, slider, textctrl):
        textctrl.SetValue(str(slider.GetValue()))

    def _tcUpdate(self, event, slider, textctrl):
        slider.SetValue(int(textctrl.GetValue()))

    def SetValue(self, x: int):
        self.slider.SetValue(x)
        self._sliderUpdate(None, self.slider, self.textControl)

    def GetValue(self):
        return self.slider.GetValue()
#---------------------------------------------------------------------------

class MyFrame(wx.Frame):
    def __init__(self, title):
        wx.Frame.__init__(self, None, -1,
                          title,
                          size=(740, 740))


        #------------
        self.SetMinSize((740, 740))

        #------------
#        dir_icons = ""
#        frameIcon = wx.Icon(os.path.join(dir_icons, "wxwin.ico"),
#                            type=wx.BITMAP_TYPE_ICO)
#        self.SetIcon(frameIcon)
    


        panels = {}
        sizer = wx.BoxSizer(wx.VERTICAL)
        sp = wx.SplitterWindow(self)
        panels = []
        for fig_name in ["left", "right"]:
            p = MyCanvasPanel(sp)
            p.plot()

            panels.append(p)
        sp.SplitVertically(*panels, 100)
        sizer.Add(sp)

        for label in ["x", "y", "z"]:
            p = IntSliderTextCtrl(self, -100, 100, 0, label=label)
            sizer.Add(p)

        for label in ["roll", "pitch", "yaw"]:
            p = IntSliderTextCtrl(self, -180, 180, 0, label=label)
            sizer.Add(p)


        #self.panel = MyCanvasPanel(self)
        #self.panel.plot()

        self.SetSizer(sizer)
        self.Fit()


class MyApp(wx.App):
    def OnInit(self):

        frame = MyFrame("Stewart Platform Test")
        self.SetTopWindow(frame)
        frame.Show(True)



        return True


def main():
    app = MyApp(False)
    app.MainLoop()


if __name__ == "__main__" :
    main()