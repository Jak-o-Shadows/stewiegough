"""

Link : https://white-wheels.hatenadiary.org/entry/20100327/p5

"""

import os
import sys
import functools

import numpy as np
import matplotlib as mpl
import matplotlib  # TODO: Remove
import matplotlib.pyplot as plt
matplotlib.interactive(True)
#matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar2Wx
from mpl_toolkits.mplot3d import Axes3D

import wx
import wx.lib.agw.floatspin

import objectBased
import vis


def on_move(fig1, ax1, fig2, ax2, event):
    if event.inaxes == ax1:
        ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    elif event.inaxes == ax2:
        ax1.view_init(elev=ax2.elev, azim=ax2.azim)
    else:
        return
    fig1.canvas.draw()
    fig2.canvas.draw()

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
        self.ax = Axes3D(self.figure, proj_type='ortho')

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.RIGHT| wx.TOP | wx.GROW)

        self.SetSizer(sizer)
        self.Fit()





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


class FloatSliderTextCtrl(wx.Panel):
    def __init__(self, parent, lower, upper, start, increment, label=None):
        wx.Panel.__init__(self, parent)

        self.parent = parent
        vbox = wx.BoxSizer(wx.HORIZONTAL)

        if label: label = wx.StaticText(self, label=label)

        self.slider = wx.Slider(self, -1, start, lower, upper, style=wx.SL_HORIZONTAL)  # TODO: Make support floats
        self.spinner = wx.lib.agw.floatspin.FloatSpin(self, -1, min_val=lower, max_val=upper, increment=increment, value=start, agwStyle=wx.lib.agw.floatspin.FS_LEFT)
        self.spinner.SetFormat("%f")
        self.spinner.SetDigits(3)

        self.slider.Bind(wx.EVT_SLIDER, lambda event: self._sliderUpdate(event, self.slider,self.spinner))
        self.spinner.Bind(wx.EVT_SPINCTRL, lambda event: self._spinnerUpdate(event, self.slider,self.spinner))

        if label: vbox.Add(label, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        vbox.Add(self.slider, 1, wx.EXPAND | wx.TOP | wx.RIGHT | wx.LEFT | wx.BOTTOM)
        vbox.Add(self.spinner, 1, wx.TOP | wx.BOTTOM)

        self.SetValue(start)


        self.SetSizer(vbox)
        self.Fit()

    def _sliderUpdate(self, event, slider, spinner):
        spinner.SetValue(slider.GetValue())

    def _spinnerUpdate(self, event, slider, spinner):
        slider.SetValue(int(spinner.GetValue()))  # TODO: Should not convert to int

    def SetValue(self, x: float):
        self.spinner.SetValue(x)
        self._spinnerUpdate(None, self.slider, self.spinner)

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
    


        self.stewart = objectBased.RotaryStewartPlatform()



        self.panels = {}
        sizer = wx.BoxSizer(wx.VERTICAL)
        sp = wx.SplitterWindow(self)
        self.panels = []
        for fig_name in ["left", "right"]:
            p = MyCanvasPanel(sp)
            self.panels.append(p)
        sp.SplitVertically(*self.panels, 100)
        sizer.Add(sp)

        # Connect the figures so they rotate as one
        update_func = functools.partial(on_move,
                                        self.panels[0].figure, self.panels[0].ax,
                                        self.panels[1].figure, self.panels[1].ax)
        self.panels[0].canvas.mpl_connect("motion_notify_event", update_func)
        self.panels[1].canvas.mpl_connect("motion_notify_event", update_func)

        self.sliders = {}
        for label in ["x", "y", "z"]:
            p = IntSliderTextCtrl(self, -100, 100, 0, label=label)
            self.sliders[label] = p
            sizer.Add(p)

        for label in ["roll", "pitch", "yaw"]:
            p = IntSliderTextCtrl(self, -30, 30, 0, label=label)
            self.sliders[label] = p
            sizer.Add(p)

        # Add refresh and add button
        self.from_pos_b = wx.Button(self, label="From Pos")
        self.from_angle_b = wx.Button(self, label="From Leg Angle")
        self.update_b = wx.Button(self, label="Update Plots")
        self.from_pos_b.Bind(wx.EVT_BUTTON, self.from_pos)
        self.from_angle_b.Bind(wx.EVT_BUTTON, self.from_angles)
        self.update_b.Bind(wx.EVT_BUTTON, self.plots_update)

        sizer.Add(self.from_pos_b)
        sizer.Add(self.from_angle_b)
        sizer.Add(self.update_b)

        # Add leg measurements
        self.lower_legs_angle_st = []
        for leg_num in range(6):
            label = f"Leg {leg_num}"
            p = FloatSliderTextCtrl(self, -100, 100, 0, 0.01, label=label)
            self.lower_legs_angle_st.append(p)
            sizer.Add(p)


        self.SetSizer(sizer)
        self.Fit()

    def plots_update(self, event):
        # Plot the simple platform on the left hand plot
        p1 = self.panels[0]
        p1.ax.clear()
        vis.plotPlatform(p1.ax, self.stewart.bPos, 'ko-')  # Base Platform
        vis.plotPlatform(p1.ax, self.stewart.trans, 'ko-')  # Upper Platform
        vis.drawLinks(p1.ax, self.stewart.bPos, self.stewart.trans, 'k--')

        p1.ax.set_xlabel('x')
        p1.ax.set_ylabel('y')
        p1.ax.set_zlabel('z')

        # Plot with actuator links on the right hand plot
        p2 = self.panels[1]
        p2.ax.clear()
        vis.plotPlatform(p2.ax, self.stewart.bPos, 'ko-')  # Base Platform
        vis.plotPlatform(p2.ax, self.stewart.trans, 'ko-')  # Upper Platform

        for i, p in enumerate(self.stewart.midJoint):
            p2.ax.plot([p[0]], [p[1]], [p[2]], 'ro')
            x= p[0]
            y= p[1]
            z= p[2]
            p2.ax.text(x, y, z, str(i), size=20, zorder=-1, color='k')

        vis.drawLinks(p2.ax, self.stewart.bPos, self.stewart.midJoint, 'r-')
        vis.drawLinks(p2.ax, self.stewart.midJoint, self.stewart.trans, 'g-')


    def from_pos(self, event):
        trans_init = [self.sliders[dimension].GetValue()/1000 for dimension in ["x", "y", "z"]]  # mm to metres
        angles_init_rad = list(np.deg2rad([self.sliders[dimension].GetValue() for dimension in ["roll", "pitch", "yaw"]]))
        self.stewart.inverse(trans_init, angles_init_rad)

        # Put the leg angles in
        for leg_num, angle_rad in enumerate(self.stewart.leverAngles):
            self.lower_legs_angle_st[leg_num].SetValue(np.rad2deg(angle_rad))


        self.plots_update(event)


    def from_angles(self, event):
        leg_angles_rad = np.rad2deg([x.GetValue() for x in self.lower_legs_angle_st])
        self.stewart.forward(leg_angles_rad)

        self.plots_update(event)




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