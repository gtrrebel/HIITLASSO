from lasso_utils import *

np.set_printoptions(precision=2)

print "start"
plot_performances(data_type='band', data_params = [], methods=["naive", "glasso", "clime", "scio"])
print "end"