#!/usr/bin/env python
# encoding: utf-8


import os.path

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on the give port", type=int)

theta = 500    #for display purpose
base = -0    #for display purpose , lager means brighter
color_mod = 1.0
normalize = True
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        sentences, words, targets = get_words()
        weights, ty, py = get_weights()
        if(normalize):        #normalize
            w_tmp_list = []
            for w in weights:
                w_tmp = [(i-min(w))/(max(w)-min(w)) for i in w]   #do some mormalize for display purpose
                w_tmp_list.append(w_tmp)
            weights = w_tmp_list
        print (weights)
        self.render('index.html', sentences=sentences, words=words, weights=weights, targets=targets, ty=ty, py=py)


def get_words():
    sentences = []
    words = []
    targets = []
    #lines = open(os.path.join(os.path.dirname(__file__), 'test.txt')).readlines()
    lines = open(os.path.join(os.path.dirname('D://Sentiment Classification//corpus_clean//'), 'Restaurants_test_v2.csv')).readlines()
    for i in range(0, len(lines), 3):
        sentences.append(lines[i].strip())
        words.append(lines[i].split())
        targets.append(lines[i + 1].split())
    #print (words[0])
    return sentences[:50], words[:50], targets[:50]


def get_weights():
    weights = []
    ty = []
    py = []
    #lines = open(os.path.join(os.path.dirname(__file__), 'weight.txt')).readlines()
    lines = open(os.path.join(os.path.dirname('C://Users//panlu//PycharmProjects//TD-LSTM//data//restaurant//'), 'alpha_out.txt')).readlines()
    #lines = open('weight.txt').readlines()
    for line in lines:
        ws = line.split()

        ty.append(ws[0])
        py.append(ws[1])
        tmp = []
        i = 0
        ws = ws[2:]       #第一个是target 第二个是predict
        for w in ws:
            if float(w) != 0:
                # tmp.append([0, i, float(w)])
                tmp.append(float(w))
                i += 1
        weights.append(tmp)
    return weights[:50], ty[:50], py[:50]

class Sentence(tornado.web.UIModule):
    def embedded_css(self):
        return 'dl {display: block; background-color: #e6e6e6;} dd {display: inline-block; padding: 0 11px; height: 26px; font-size: 14px; line-height: 26px; margin:0 5px 5px 0;}'

    def render(self, number, sentence, word, weight, target, ty, py):
        html = ''
        html += '<dl>target:%s, true y: %s, predict y:%s</br></br>' % (' '.join(target), ty, py)
        for wd, wt in zip(word, weight):
            wt = 99-int(wt * theta) + base
            wt = int(wt * color_mod)
            #print (wt)
            wt = '#12' + str(wt) + '90'
            html += ('<dd style="%s">%s</dd>' % ("background-color: " + wt, wd))
            # wt = max(int(wt * 100), 10)
            # html += ('<dd style="%s">%s</dd>' % ("font-size: " + str(wt) + 'px;', wd))
        html += '</dl>'
        return html


class Chart(tornado.web.UIModule):
    def render(self, number, sentence, word, weight, target, ty, py):
        return self.render_string("chart.html", number=str(number), sentence=sentence, word=word, weight=weight, target=target, ty=ty, py=py)


if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[(r'/', IndexHandler),
            ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        ui_modules={
            'Chart': Chart,
            'Sentence': Sentence,
        },
        debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


