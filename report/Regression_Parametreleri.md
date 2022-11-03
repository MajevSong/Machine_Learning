-- Linear Regression ---

Doğrusal problemler için kullandığımız model.

class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

fit_intercept => LinearModel için uygun bir kesişim hesaplanıp hesaplanmayacağını belirliyoruz. Eğer kullanılmak istenmiyorsa False olarak belirtilir.
ama default olarak True olarak ayarlanır.
copy_X => eğer True ise X kopyalanacak aksi takdir de üstüne yazılacak. Default olarak True olarak ayarlıdır
n_jobs => Hesaplama için kullanılacak iş sayısı. Bu yalnızca yeterince büyük sorunlar olması durumunda hızlanma sağlayacaktır. Default olarak None kullanılır.
positive => Eğer true olarak ayarlanırsa katsayıları pozitif olmaya zorlar. Bu seçenek yalnızca yoğun diziler için desteklenir.

--- Polynomial Regression ---

Verilerin üstel şekilde dağıldığı dizilimlerde kullanılan bir modeldir.

class sklearn.preprocessing.PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order='C')

degree => Polinomun derecesini belirtiyoruz. Default olarak 2'dir
interaction_only => Eğer true yapılırsa inputlar derecesi 1 olarak düşünülüp yalnızca çarpılır.
include_bias => Eğer true ise polynomial üstleri sıfır olan tüm özellikler için bir sapma(bias) kolonu ekleriz. Bu sapma kolonu 1'lerden oluşur. 
order => Yoğun durumda çıktı dizisinin sırasını belirtir. Default olarak C kullanılır. Ama 'F' seçilecek olursa sıralama daha hızlı hesaplanabilir ama sonraki tahminleri yavaşlatabilir.

--- Support Vector Regression ---

SVR vektörü marjinel verilere karşı çok hassastır. İlk önce ölçeklendirme yapıldıktan sonra kullanılır.

class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

kernel => SVR hesaplamasını bu çekirdek modele göre inşa eder. Bunlar;
- linear 
- poly
- rbf
- sigmoid
- precomputed
degree => Eğer çekirdek "poly" seçildiyse polinomun derecesini belirtmek için kullanırız.
gamma => "rbf", "poly" ve "sigmoid" için çekirdek katsayısı. gamma="scale" seçildiyse o zaman gammanın değeri olarak 1/(n_features*X.var()) formülü kullanılır.
şayet "auto" seçilirse "1/n_features" formülü uygulanır. 
coef0 => Çekirdek functiondan bağımsız bir terimdir. Yanlızca "poly" ve "sigmoid" için önemlidir.
tol => Durdurma kriteri için tolerans belirliyoruz. Default ayarı 1e-3
cache_size => Çekirdek önbelleğinin boyutunu belirtir.
verbose => Ayrıntılı çıktı verir.
max_iter => iterasyon sayısının limitini belirlemek için kullanılır. Eğer -1 verilirse sınırsız iterasyon kullanılır.

--- Decision Tree ---

Bölerek sınıflandıran bir algoritmadır. Bu algoritma da en büyük sıkıntı overfitting (aşırı öğrenme) den kaynaklı sorundur. Çok fazla bölme işlemi gerçekleştirilirse ezberleme ihtimali artabilir.

class sklearn.tree.DecisionTreeRegressor(*, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)

criterion => Yaptığı bölme işleminin kalitesini ölçmek için kullanılır.
splitter => Her düğümde bölümü yapmak için kullanacağı stratejiyi seçiyoruz. "Best" veya "random" seçebiliriz
max_depth => Ağaç yapısının maksimum derinliğini belirtiriz. Eğer none seçilirse tüm yapraklar saf kalana kadar genişletilir.
min_samples_split => Bir düğümü bölmek için minimum örnek sayısını belirtir. Default olarak 2 dir
min_samples_leaf => Bir yaprak düğümde olması gereken minimum örnek sayısı.
min_weight_fraction_leaf => Bir yaprak düğümde olması gereken ağırlıkların toplamının minimum ağırlıklı kesri
max_features => En iyi bölünmeyi ararken göz önünde bulundurulması gereken özelliklerin sayısı
random_state => Tahminin rastgeleliğini kontrol eder
max_leaf_nodes => Maks yaprak düğümünü belirtiyoruz. 
ccp_alpha => Minimum karmaşıklık matrisi

--- Random Forest ---

Birden fazla Decision Tree oluşturularak overfitting aşılmaya çalışılır. Farklı farklı decision treeler ile bir öğrenme gerçekleştirir.

class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

n_estimators => kaç ağaçdan oluşacağını belirtiyoruz.
** diğer parametreleri decision tree ile benzer.




