import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from PIL import Image
import glob
import base64
import re
import pandas as pd
import string

from predict import test

app = dash.Dash(__name__)
server = app.server
list_of_images = ["real.png","fake.png"]
encoded_real_image = base64.b64encode(open(list_of_images[0], 'rb').read())
encoded_fake_image = base64.b64encode(open(list_of_images[1], 'rb').read())
encoded_school_image = base64.b64encode(open("NYCU_logo.png", 'rb').read())
real_source='data:image/png;base64,{}'.format(encoded_real_image.decode())
fake_source='data:image/png;base64,{}'.format(encoded_fake_image.decode())
school_source='data:image/png;base64,{}'.format(encoded_school_image.decode())
lis=[html.Button('Refresh', id='refresh', n_clicks=0, style={"width":"100px", "height":"40px", "font-size":"16px", "margin":"0px 0px 0px 305px"}),html.Br(),]
for i in range(10):
    lis.append(html.H4('', style={"width":"50","height":"150","font-size":"20px",'display': 'inline-block','verticalAlign':'middle'}))
    lis.append(html.Button('news_sample', value='news_sample', id='news'+str(i) , style={"width":"500px", "height":"150px", "font-family":"Times New Roman", "font-size":"24px", "overflow": "hidden",'background-color': 'white','border': '2px solid #C2C287','border-radius': '8px',"margin":"5px 0px 0px 0px",'display': 'inline-block'}))

app.layout = html.Div([
    html.Img(id="school_image",src=school_source, style={"textAlign": "center", "width":"250px", "height":"50px","margin":"0px 0px 0px 0px"}),
    html.H1("Fake News Detection Demo Web", style={"textAlign": "center", "background": "#F5FFE8","margin":"0px 0px 0px 0px"}),
    html.H5("Advanced Database System Laboratory", style={"textAlign": "center", "background": "#F5FFE8","margin":"0px 0px 30px 0px"}),
    html.Div([
        html.H4("Input News Content: ", style={"font-size":"20px"}),
        html.Div([
            dcc.Textarea(
                id="news-input",
                value="input your news content here",
                style={"width":"800px", "height":"350px", "font-size":"18px", "max-width":"800px", "max-height":"350px"},
            ),
            html.Div(id='news-output', style={"width":"800px", "height":"350px", "font-size":"18px", "whiteSpace": "pre-wrap", "display":"none", "overflow": "scroll"}),
            html.Div(id='notice',children=[
                html.H5("Red Words",style={'color': 'red','font-size':'18px','display': 'inline-block',"margin":"5px 0px 0px 7px"}),
                html.H5("are corresponding to the",style={'color': 'black','font-size':'18px','display': 'inline-block',"margin":"5px 0px 0px 6px"}),
                html.H5("Red Node",style={'color': 'red','font-size':'18px','display': 'inline-block',"margin":"5px 0px 0px 7px"}),
                html.H5("<UNK> in the word graph below.",style={'color': 'black','font-size':'18px','display': 'inline-block',"margin":"5px 0px 0px 6px"})
            ],style={'display':'none'})
        ]),
        html.Br(),
        html.Div([
            html.H4("number of nodes in word graph = ", style={"width":"300px","margin":"5px 0px 0px 0px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'}),
            dcc.Dropdown(
                id='node_number',
                options=[
                    {'label':'50','value':'50'},
                    {'label':'250','value':'250'},
                    {'label':'500','value':'500'},
                    {'label':'1000','value':'1000'},
                    {'label':'10000','value':'10000'},
                ],
                value='10000',
                style={"width":"100px","margin":"0px 0px 0px 0px",'display':'inline-block','verticalAlign':'top'},
            ),
            html.H4(", accuracy: ", style={"width":"100px","margin":"5px 0px 0px 5px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'}),
            html.Div(id='accuracy',children=[html.H4("0.9623", style={"width":"100px","margin":"5px 0px 0px 0px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'})],style={"width":"100px","margin":"0px 0px 0px 0px",'display':'inline-block','verticalAlign':'top'}),
        ], style={"margin":"0px 0px 0px 150px"}),
        html.Br(),
        html.Button('Submit', id='submit', n_clicks=0, style={"width":"100px", "height":"40px", "font-size":"16px", "margin":"0px 0px 0px 295px", 'display': 'inline-block'}),
        html.Button('Restart', id='restart', n_clicks=0, style={"width":"100px", "height":"40px", "font-size":"16px", "margin":"0px 0px 0px 10px", 'display': 'inline-block'}),
        html.H4("Model Prediction (✓: real news, ✗: fake news): ", style={"font-size":"20px"}),
        html.Div(children=[
            html.Img(id="predict_image",src=None, style={"width":"300px", "height":"300px"}),
        ], style={'display': 'inline-block',"margin":"0px 0px 50px 0px"}),
        html.Div(children=[
            html.Img(id="word_graph_image",src=None, style={"width":"800px", "height":"300px"}),
        ], style={'display': 'inline-block'})
    ],style={"width":"1000px", "height":"1500px",'display': 'inline-block','verticalAlign':'top'}),
    html.Div(id='sample_input',children=lis,style={"width":"700px", "height":"1500px",'display': 'inline-block','verticalAlign':'top', "overflow": "scroll"})
])

@app.callback(Output('sample_input','children'),Input('refresh','n_clicks'))
def sample(n_clicks):
    lis=[html.Button('Refresh', id='refresh', n_clicks=0, style={"width":"100px", "height":"40px", "font-size":"16px", "margin":"0px 0px 0px 305px"}),html.Br(),]
    df_data = pd.read_csv('dataset/corpusSources.tsv', sep='\t', encoding='utf-8')
    valid_data = ~df_data['content'].isna()
    df_data = df_data[valid_data][['Non-credible', 'content']]
    df_data =df_data.sample(n=10)
    label=df_data['Non-credible']
    content = df_data['content'].str.lower()
    content = content.str.replace('[{}]'.format(string.punctuation), ' ')
    content = content.str.replace('\s+', ' ', regex=True)
    color_list=["#EFFFD7","#FFF8D7","#FFEEDD","#ECFFFF"]
    for index,(i,j) in enumerate(zip(content,label)):
        if j==0:
            lis.append(html.H4("Real(✓)", style={"width":"100","height":"150","font-size":"20px",'display': 'inline-block', "margin":"0px 25px 0px 25px",'verticalAlign':'middle'}))
        else:
            lis.append(html.H4("Fake(✗)", style={"width":"100","height":"150","font-size":"20px",'display': 'inline-block', "margin":"0px 25px 0px 25px",'verticalAlign':'middle'}))
        lis.append(html.Button(i, value=i, id='news'+str(index), n_clicks=0, style={"width":"500px", "height":"150px", "font-family":"Times New Roman", "font-size":"24px", "overflow": "hidden",'background-color': 'white','border': '2px solid #C2C287','border-radius': '8px',"margin":"5px 0px 0px 0px",'display': 'inline-block','verticalAlign':'middle'}))
        lis.append(html.Br())
    return lis

@app.callback([Output('news'+str(i),'n_clicks') for i in range(10)],[Output('news'+str(i),'style') for i in range(10)],Output('submit','n_clicks'), Output('restart','n_clicks'), Output('accuracy','children'), Output('notice','style'), Output('news-input','value'), Output('news-input','style'), Output('news-output','children'), Output('news-output','style'),Output('predict_image', 'src'), Output('predict_image', 'style'), Output('word_graph_image', 'src'), Output('word_graph_image', 'style'), Input('submit','n_clicks'), Input('news-input','value'), Input('restart','n_clicks'),  Input('node_number','value'), [Input('news'+str(i),'n_clicks') for i in range(10)], [Input('news'+str(i),'value') for i in range(10)])
def submit_input(submit_n_clicks,value,restart_n_clicks,node_value, *args):
    print(submit_n_clicks,restart_n_clicks)
    sample_clicked=False
    index_clicked=-1
    accuracy=0
    if node_value=='50':
        accuracy=0.7595
    elif node_value=='250':
        accuracy=0.8754
    elif node_value=='500':
        accuracy=0.9133
    elif node_value=='1000':
        accuracy=0.9260
    else:
        accuracy=0.9623
        
    for i in range(10):
        if args[i]:
            sample_clicked=True
            index_clicked=i
    color=[]
    for i in range(10):
        if i==index_clicked:
            color.append({"width":"500px", "height":"150px", "font-family":"Times New Roman", "font-size":"24px", "overflow": "hidden",'background-color': '#F5FFE8','border': '2px solid #C2C287','border-radius': '8px',"margin":"5px 0px 0px 0px",'display': 'inline-block','verticalAlign':'middle'})
        else:
            color.append({"width":"500px", "height":"150px", "font-family":"Times New Roman", "font-size":"24px", "overflow": "hidden",'background-color': 'white','border': '2px solid #C2C287','border-radius': '8px',"margin":"5px 0px 0px 0px",'display': 'inline-block','verticalAlign':'middle'})
    if submit_n_clicks or sample_clicked:
        text=value
        if submit_n_clicks:
            print("submit")
            text=re.sub("[\n]"," ",value)
            text=re.sub("[^a-zA-Z' ]","",text)
            input_value=value
        elif args[:10]:
            print("sample")
            text=re.sub("[\n]"," ",args[10+index_clicked])
            text=re.sub("[^a-zA-Z' ]","",text)
            input_value=args[10+index_clicked]
        nodes,predict=test(text.lower(),node_value)
        encoded_word_graph_image = base64.b64encode(open("word_graph.png", 'rb').read())
        word_graph_source='data:image/png;base64,{}'.format(encoded_word_graph_image.decode())
        color_txt=[]
        index=0
        for word in input_value.strip().split():
            has_alpha=False
            for alpha in word:
                if alpha.isalpha()==True:
                    has_alpha=True
            if has_alpha==True:
                if nodes[0][index]==0:
                    color_txt.append(html.H4(word, style={'color': 'red','display': 'inline-block', "font-size":"18px","margin":"5px 0px 0px 7px"}))
                else:
                    color_txt.append(html.H4(word, style={'display': 'inline-block', "font-size":"18px","margin":"5px 0px 0px 7px"}))
                index+=1
            else:
                color_txt.append(html.H4(word, style={'display': 'inline-block', "font-size":"18px","margin":"5px 0px 0px 7px"}))
        
        if predict==0:
            return 0,0,0,0,0,0,0,0,0,0,color[0],color[1],color[2],color[3],color[4],color[5],color[6],color[7],color[8],color[9],0,0,[html.H4(accuracy, style={"width":"100px","margin":"5px 0px 0px 0px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'})],{'display':'block'},"input your news content here",{'display':'none'}, color_txt, {"width":"800px", "height":"350px", "font-size":"18px", "whiteSpace": "pre-line", 'display':'block', "overflow": "scroll"}, real_source, {'display':'block','width':'300px', 'height':'300px'}, word_graph_source, {'display':'block','width':'600px', 'height':'400px'}
        elif predict==1:
            return 0,0,0,0,0,0,0,0,0,0,color[0],color[1],color[2],color[3],color[4],color[5],color[6],color[7],color[8],color[9],0,0,[html.H4(accuracy, style={"width":"100px","margin":"5px 0px 0px 0px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'})],{'display':'block'},"input your news content here",{'display':'none'}, color_txt, {"width":"800px", "height":"350px", "font-size":"18px", "whiteSpace": "pre-line", 'display':'block', "overflow": "scroll"}, fake_source, {'display':'block','width':'300px', 'height':'300px'}, word_graph_source, {'display':'block','width':'600px', 'height':'400px'}
    elif restart_n_clicks:
        print("restart")
        return 0,0,0,0,0,0,0,0,0,0,color[0],color[1],color[2],color[3],color[4],color[5],color[6],color[7],color[8],color[9],0,0,[html.H4(accuracy, style={"width":"100px","margin":"5px 0px 0px 0px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'})],{'display':'none'},"input your news content here",{"width":"800px", "height":"350px", "font-size":"18px", "max-width":"800px", "max-height":"350px", 'display':'block'}, "", {'display':'none'}, None,{'display':'none'}, None, {'display':'none'}
    else:
        print("initial")
        return 0,0,0,0,0,0,0,0,0,0,color[0],color[1],color[2],color[3],color[4],color[5],color[6],color[7],color[8],color[9],0,0,[html.H4(accuracy, style={"width":"100px","margin":"5px 0px 0px 0px","font-size":"20px", 'display':'inline-block','verticalAlign':'top'})],{'display':'none'},value,{"width":"800px", "height":"350px", "font-size":"18px", "max-width":"800px", "max-height":"350px", 'display':'block'}, value, {"display":"none"}, None, {'display':'none'}, None, {'display':'none'}

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')