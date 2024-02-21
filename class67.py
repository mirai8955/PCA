import sklearn.datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy
from factor_analyzer import FactorAnalyzer

#dw = sklearn.datasets.load_iris() # Iris data
#dw = sklearn.datasets.load_boston()
#dw = sklearn.datasets.load_diabetes()
#dw = sklearn.datasets.load_digits()
#dw = sklearn.datasets.load_linnerud()
dw = sklearn.datasets.load_wine()

print('\n', 'Class67.py\n')
print(dw['target_names'])  # Class names
print(dw['feature_names']) # Variable names
#print(dw.data)
#print(dw.target)

features = dw.data
print(features.shape, '\n')

##
# Ourlier detection using Mahalanobis's generalized distance
##

cov = np.cov( np.transpose(features) )
print( 'Covariance matrix =')
print( cov )
print()

inv_cov = np.linalg.inv( cov )
print( 'Inverse of covariance matrix =' )
print( inv_cov, '\n' )

center = np.mean( features, axis=0 )
print( 'Centroid =', center, '\n' )

d = [0] * 178

for i in range(178):
    x = features[i,:]
    dim = len(x)
    d_square = np.transpose(x - center) @ inv_cov @ (x - center)#まはらのびす距離の2上を計算
    #print( x )
    #print( x - center )
    #print( d_square**0.5 )
    d[i] = d_square**0.5
    if d_square > scipy.stats.distributions.chi2.ppf( 0.99, dim ): # (1-0.99) is the significance level. Please, use 0.99 or 0.95.ここで外れ値の判定
        print( 'Outlier[%d]: d_square = %f' % (i, d_square ) )
        print( ' ' + str(x) )

# 6つの外れ値が見つかり、除外しました．中身をよく見ずに除外しました．外れ値として除外すべきか十分に検討するところです
# 外れ値の基準として 0.99 を用いました．0.95 でも構いません．
new_features = np.delete( features, [69, 73, 95, 110, 121, 158], axis=0 )

#
# PCA/FA
#
zfeatures = scipy.stats.zscore( new_features ) # Standardize (z-score)
#print()
#print( np.mean( zfeatures, axis=0 ) )
#print( np.var( zfeatures, axis=0 ) )

# ここでは主成分の数を変数の数と同じにしていますが、それは固有値を見て、いくつの主成分を採用するか検討するためです
pca = PCA( n_components=dim )
pca.fit( zfeatures )

#print()
#print( pca.mean_ )

#print( '\n-PC coefficients' )
#print( pca.components_ )

print( '\n-Contribution ratios' )
print( pca.explained_variance_ratio_ )
plt.plot( pca.explained_variance_ratio_*dim, "o-" )
plt.title( "Scree graph" )
plt.ylabel( "Eigen value" )
plt.show()

# このサンプルでは，Kayser-Gutman 基準を用いて λ >= 1 となる３つの主成分を採用しました
# 累積寄与率やスクリーグラフの形を基準としてもらっても構いません

pcscores = pca.transform( zfeatures )

#plt.subplot(121)
#plt.scatter( pcscores[:,0], pcscores[:,1] )
#plt.xlabel( "1st PC" )
#plt.ylabel( "2nd PC" )

#plt.subplot(122)
#plt.scatter( pcscores[:,0], pcscores[:,2] )
#plt.xlabel( "1st PC" )
#plt.ylabel( "3rd PC" )

#plt.show()

#
# FA - Rotation
#

# Python の PCA には回転の機能が実装されていません．回転を使うために因子分析 FactorAnalyzer のパッケージを使っています．
# PCA と FA はよく似ていますので，ここでは FA で PCA を代用しています．
#
fap = FactorAnalyzer( n_factors=3, rotation='promax' ) # promax法を使う場合がこちらです -> fap
fan = FactorAnalyzer( n_factors=3, rotation=None ) # 回転させない場合がこちらです -> fan
fap.fit( zfeatures ) # FAの計算
fan.fit( zfeatures )
fapscores = fap.transform( zfeatures ) # FA得点の計算
fanscores = fan.transform( zfeatures )

eigpa, eigpb = fap.get_eigenvalues()
eigna, eignb = fan.get_eigenvalues()

print()
print( eigna ) # eigenvalues
#print( eignb ) # eigenvalues after removing sample errors
print()
print("-Loadings before rotation-")
print( fan.loadings_ )
print()
print("-Loadings after rotation-")
print( fap.loadings_ )

xax = range(0,dim)
plt.title( "Loadings (PC coefficients)" )
#plt.plot( xax, pca.components_[:,0:3], "b-" )
plt.plot( xax, fan.loadings_[:,0], "r-", label="1st PC" ) # PC coefficients before rotation
plt.plot( xax, fan.loadings_[:,1], "b-", label="2nd PC" )
plt.plot( xax, fan.loadings_[:,2], "g-", label="3rd PC" )
plt.plot( xax, fap.loadings_[:,0], "r--", label="1st PC after rotation" ) # PC coefficients after rotation
plt.plot( xax, fap.loadings_[:,1], "b--", label="2nd PC after rotation" )
plt.plot( xax, fap.loadings_[:,2], "g--", label="3rd PC after rotation" )
plt.legend()
plt.show()

# この例題では，回転によって PC係数（負荷量）はあまり変化しませんでしたが，大きく変化することもあります．
# 回転の前後で，主成分を解釈して下さい

#print( np.linalg.norm( pca.components_[:,0], 2 ) )
#print( np.linalg.norm( fan.loadings_[:,0], 2 )**2 )
#print( np.linalg.norm( fan.loadings_[:,1], 2 )**2 )
#print( np.linalg.norm( fan.loadings_[:,2], 2 )**2 )

plt.subplot(221)
plt.scatter( fanscores[:,0]*eignb[0]**.5, fanscores[:,1]*eignb[1]**.5 )
plt.title( "Before rotation ")
#plt.xlabel( "1st PC" )
plt.ylabel( "2nd PC" )

plt.subplot(222)
plt.scatter( fanscores[:,0]*eignb[0]**.5, fanscores[:,2]*eignb[2]**.5 )
plt.title( "Before rotation ")
#plt.xlabel( "1st PC" )
plt.ylabel( "3rd PC" )

plt.subplot(223)
#plt.scatter( pcscores[:,0], pcscores[:,1] )
plt.scatter( fapscores[:,0], fapscores[:,1] )
plt.title( "After rotation ")
plt.xlabel( "1st PC" )
plt.ylabel( "2nd PC" )

plt.subplot(224)
#plt.scatter( pcscores[:,0], pcscores[:,2] )
plt.scatter( fapscores[:,0], fapscores[:,2] )
plt.title( "After rotation ")
plt.xlabel( "1st PC" )
plt.ylabel( "3rd PC" )

plt.show()

#print( np.mean( fascores[:,0]) )
#print( np.var( fascores[:,0]) )
#print( np.mean( fascores[:,1]) )
#print( np.var( fascores[:,1]) )
#print( np.mean( fascores[:,2]) )
#print( np.var( fascores[:,2]) )