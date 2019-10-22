import ipywidgets as widgets
from ipywidgets import FloatSlider, HBox, Button, Output, FloatText, VBox, BoundedFloatText
from ipywidgets import Layout,Label,interactive_output, interactive
from numpy.linalg import norm, det, inv
import numpy as np
from Tenuous import TenuousModel

# Define global parameters for parameter checks
params_pass = False
model_solved = False

style_mini = {'description_width': '0px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '200px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='30%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')

qₒₛ = BoundedFloatText(min=0, max=0.2, step=0.01, value=0.1, description = r'$q_{s,0}$', style=style_med,layout = Layout(width='70%'))
qu = BoundedFloatText(min=0, max=0.2, step=0.01, value=0.1, description = r'$q_{u,s}$', style=style_med,layout = Layout(width='70%'))
RestrictedRho = widgets.Dropdown(
    options = {'Yes', 'No'},
    value = 'Yes',
    description=r'Apply restriction to $\rho_{2}$',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

ρ2 = BoundedFloatText(min=0.1, max=1, step=0.1, value=1, description='ρ2 Multiplier', style=style_med,layout = Layout(width='70%'))
ρ1 = BoundedFloatText(min=0, max=1, step=0.1, value=0, description='ρ1', style=style_med,layout = Layout(width='70%'))

δ = FloatText(min=0, max=0.1, step=0.001, value=0.002, description='δ (Discount rate)', style=style_med,layout = Layout(width='70%'))
sigk1 = FloatText(min=0, max=0.5, step=0.001, value=0.477, description=r"$\sigma_{k1}$", style=style_med,layout = Layout(width='70%'))
sigk2 = FloatText(min=0, max=0.5, step=0.001, value=0, description=r"$\sigma_{k2}$", style=style_med,layout = Layout(width='70%'))

sigz1 = FloatText(min=0, max=0.05, step=0.001, value=0.011, description=r"$\sigma_{z1}$", style=style_med,layout = Layout(width='70%'))
sigz2 = FloatText(min=0, max=0.05, step=0.001, value=0.025, description=r"$\sigma_{z2}$", style=style_med,layout = Layout(width='70%'))

αŷ =  FloatText(min=0, max=0.5, step=0.001, value=0.484, description=r'${\widehat \alpha}_k $', style=style_med,layout = Layout(width='70%'))
αẑ̂ =  FloatText(min=0, max=0, step=0.001, value=0, description=r'${\widehat \alpha}_z $', style=style_med,layout = Layout(width='70%'))

β̂ = FloatText(min=0, max=1, step=0.01, value=1, description=r'${\widehat \beta}_k $', style=style_med,layout = Layout(width='70%')) # >0 <1
κ̂ = FloatText(min=0, max=0.3, step=0.01, value=0.014, description=r'${\widehat \beta}_z $', style=style_med,layout = Layout(width='70%'))

box_layout       = Layout(width='100%', flex_flow = 'row')#, justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='10%')

sigk_box1 = VBox([sigk1, sigk2], layout = Layout(width='90%'))
sigk_box = VBox([widgets.Label(value=r"$\sigma_{k}$"), sigk_box1])

sigz_box1 = VBox([sigz1, sigz2], layout = Layout(width='90%'))
sigz_box = VBox([widgets.Label(value=r"$\sigma_{z}$"), sigz_box1])

sigmas_box = VBox([sigk_box, sigz_box], layout = Layout(width='100%'))

Zbox = VBox([widgets.Label(value="Baseline Model Dynamics"), κ̂ , αẑ̂, sigz_box], layout = Layout(width='90%'))
Kbox = VBox([widgets.Label(value="Capital Dynamics"), β̂ , αŷ, sigk_box], layout = Layout(width='90%'))

entropybox = VBox([widgets.Label(value="Entropy Parameters"), qₒₛ, qu], layout = Layout(width='90%'))
rhobox = VBox([widgets.Label(value="Generator Function Parameters"), RestrictedRho, ρ2, ρ1], layout = Layout(width='90%'))

line1 = HBox([Zbox, Kbox], layout = box_layout)
line2 = HBox([entropybox, rhobox], layout = box_layout)

# left_box = VBox([Label('Capital Dynamics'), σy1, σy2, αŷ, κ̂])
# middle_box = VBox([Label('Baseline Model Dynamics'), σz1, σz2, αẑ̂ , β̂])
# right_box = VBox([Label('Entropy and other Parameters'), qₒₛ, qᵤₛ, ρ2, ρ1, δ])
# ui = HBox([left_box, middle_box, right_box])
button_update = Button(description = "Update Parameters")
button_solve = Button(description = "Solve Model")
button_plot = Button(description = "Plot")

userdefinedmodel = None

params_pass = False
model_solved = False
userparams = {}
usermodel = None

def updateparams(b):
    global userparams
    global params_pass
    global usermodel
    userparams['q'] = 0.05

    userparams['αk'] = αŷ.value #0.386
    userparams['αz'] = αẑ̂.value
    userparams['βk'] = β̂.value
    userparams['βz'] = κ̂.value 
    userparams['σy'] = np.array([[sigk1.value], [sigk2.value]])
    userparams['σz'] = np.array([[sigz1.value], [sigz2.value]])
    userparams['δ'] = 0.002

    userparams['ρ1'] = ρ1
    userparams['ρ2'] = userparams['q'] ** 2 / norm(userparams['σz']) ** 2
    userparams['z̄'] = userparams['αz'] / userparams['βz']
    userparams['σ'] = np.vstack([userparams['σy'].T, userparams['σz'].T ]) 
    userparams['a'] = norm(userparams['σz']) ** 2 /  det(userparams['σ'] ) ** 2
    userparams['b'] = - np.squeeze(userparams['σy'].T.dot(userparams['σz'])) /  det(userparams['σ'] ) ** 2
    userparams['d'] = norm(userparams['σy']) ** 2 /  det(userparams['σ'] ) ** 2

    userparams['zl'] = -2.5
    userparams['zr'] = 2.5
    
    q0s = qₒₛ.value
    qus = qu.value
    if RestrictedRho.value == True:
        rho = None
    else:
        rho = ρ2.value
    usermodel = TenuousModel(userparams, [q0s], [qus], [rho])
    params_pass = True
    print("Parameters updated")

def solvemodel(b):
    global params_pass
    global usermodel
    global model_solved
    
    if params_pass:
        usermodel.solve()
        model_solved = True
        print("Model Solved")
    else:
        print("Parameters need to be passed first")
    
def showplots(b):
    global model_solved
    
    if model_solved:
        usermodel.driftplot()
        usermodel.shockplot()
    else:
        print("Model need to be solved first")
        
button_update.on_click(updateparams)
button_solve.on_click(solvemodel)
button_plot.on_click(showplots)
