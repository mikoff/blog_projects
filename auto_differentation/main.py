from autodiff import *


########################################################
# Example 1: calculate gradients w.r.t. x and y
########################################################
cg = ComputationalGraph()
cg.add_inputs(x = 2., y = -4.)
cg.add_computation_node('sigy', Sigmoid, ('y',))
cg.add_computation_node('num', Add, ('x', 'sigy'))
cg.add_computation_node('sigx', Sigmoid, ('x',))
cg.add_computation_node('xpy', Add, ('x', 'y'))
cg.add_computation_node('sq', Square, ('xpy',))
cg.add_computation_node('denom', Add, ('sigx', 'sq'))
cg.add_computation_node('f', Div, ('num', 'denom'))

cg.forward('f')
cg.backward('f')

dx = cg.get_gradient('x')
dy = cg.get_gradient('y')
print(f'df/dx = {dx}, df/dy = {dy}')


########################################################
# Example 2: calculate gradients of f w.r.t. parameters
########################################################
cg = ComputationalGraph()
cg.add_inputs(x = 3., y = 7., w0 = -2, w1 = 2)
cg.add_computation_node('w1_times_x', Multiply, ('w1', 'x'))
cg.add_computation_node('prediction', Add, ('w0', 'w1_times_x'))
cg.add_computation_node('l', Sub, ('prediction', 'y'))
cg.add_computation_node('f', Square, ('l',))

cg.forward('f')
cg.backward('f')

dw0 = cg.get_gradient('w0')
dw1 = cg.get_gradient('w1')
print(f'df/dw0 = {dw0}, df/dw1 = {dw1}')


########################################################
# Example 3: minimization using GD & forward-backward pass
########################################################
cg = ComputationalGraph()
cg.add_inputs(x = 0., y = 0., a = 400., b = 8.)
cg.add_computation_node('sqx', Square, ('x',))
cg.add_computation_node('sqy', Square, ('y',))
cg.add_computation_node('ax', Multiply, ('a', 'x'))
cg.add_computation_node('by', Multiply, ('b', 'y'))
cg.add_computation_node('sqxpsqy', Add, ('sqx', 'sqy'))
cg.add_computation_node('axpby', Add, ('ax', 'by'))
cg.add_computation_node('f', Add, ('sqxpsqy', 'axpby'))

for i in range(0, 50):
    cg.forward('f')
    cg.backward('f')

    dx = cg.get_gradient('x')
    dy = cg.get_gradient('y')
    cg.inputs['x'].output -= dx * 0.1
    cg.inputs['y'].output -= dy * 0.1
    
print('Minimum:', cg.inputs['x'].output, cg.inputs['y'].output)
