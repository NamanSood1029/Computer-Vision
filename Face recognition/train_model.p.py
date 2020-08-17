from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--embeddings', required=True, help='path to serialized db of facial embeddings')
ap.add_argument('-r', '--recognizer', required=True, help='path to the output model trained to recognize faces')
ap.add_argument('-l', '--le', required=True, help='path to output label encoder')
args = vars(ap.parse_args())

print('[INFO] loading face embeddings...')
data = pickle.loads(open(args['embeddings'], 'rb').read())

# encode labels for predictions
print('[INFO] encoding labels...')
le = LabelEncoder()
labels = le.fit_transform(data['names'])

print('[INFO] training model...')
# recognizer = SVC(kernel = 'linear', probability=True)
# recognizer.fit(data['embeddings'], labels)
recognizer = RandomForestClassifier()
recognizer.fit(data['embeddings'], labels)
# Open first argument - path/name of fle
# second argument - r (read) w(write) t(text) b(binary) x(create) a(append)
# Writing them to disk

f = open(args['recognizer'], 'wb')
f.write(pickle.dumps(recognizer))
f.close()

f = open(args['le'], 'wb')
f.write(pickle.dumps(le))
f.close()