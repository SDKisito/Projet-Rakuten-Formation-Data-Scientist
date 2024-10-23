import fasttext as ft
import fasttext.util as ftutil

print("Load original, 300-dimensional vectors")
vecs = ft.load_model("cc.fr.300.bin")

print("Reduce vector space to 100 dimensions")
ftutil.reduce_model(vecs, 100)

words = vecs.get_words()
n_words = len(words)
step_size = 1
while step_size * 100 < n_words:
    step_size *= 10

output_filename = "cc.fr.100.vec"
print("Write 100-dimensional vectors to '{}'".format(output_filename))
with open(output_filename, "w") as file_out:
    file_out.write(str(n_words) + " " + str(vecs.get_dimension()) + "\n")
    for i_step, w in enumerate(words):
        v = vecs.get_word_vector(w)
        vstr = " ".join(str("%.4f" % vi) for vi in v)
        try:
            _ = file_out.write(w + " " + vstr+'\n')
        except:
            pass
        if (i_step + 1) % step_size == 0 or i_step + 1 == n_words:
            print("{} / {}".format(i_step + 1, n_words))
