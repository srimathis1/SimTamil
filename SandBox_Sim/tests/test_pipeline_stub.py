import sys,json,os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));sys.path.insert(0,'.')
import llm
llm.available=lambda:True
calls=[]
def fake(system,prompt,json_mode=False,temperature=.9,retries=4,max_tokens=None):
    calls.append(prompt[:40])
    if 'Return JSON: {"bio_ta"' in prompt: return json.dumps({"bio_ta":"சோதனை அறிமுகம்","speaking_style_ta":"பாணி","core_memories_ta":["நினைவு %d"%i for i in range(8)],"values_ta":["a","b"],"sample_phrases":["x"]},ensure_ascii=False)
    if 'hour' in prompt and 'Allowed places' in prompt: return '```json\n[{"hour":6,"place":"home:1","activity_ta":"தூங்குகிறார்"},{"hour":9,"place":"cafe","activity_ta":"தேநீர்"},{"hour":12,"place":"mars","activity_ta":"bad"},{"hour":14,"place":"library","activity_ta":"படிக்கிறார்"},{"hour":18,"place":"park","activity_ta":"நடை"},{"hour":21,"place":"home:1","activity_ta":"தூக்கம்"}]\n```'
    if 'நிகழ்வு (' in prompt:
        ids=[l[1:-1] for l in prompt.split('\n') if l.startswith('[') and l.endswith(']')]
        return json.dumps([{"id":i,"text":"இந்த கொள்கை பற்றி "+i,"action":"gather" if k%2 else "flee","mood":"கவலை"} for k,i in enumerate(ids[:-1])],ensure_ascii=False)  # one missing -> fallback
    if 'JSON மட்டும்: [{"s"' in prompt: return json.dumps([{"s":"Afraa","t":"வணக்கம் சூர்யா"},{"s":"Surya","t":"வணக்கம் அஃப்ரா"}],ensure_ascii=False)
    if 'Summarise' in system: return 'சுருக்கம்'
    return 'தமிழ் பதில் '+str(len(calls))
llm.call=fake
import server,memory
import time
while server.STATUS["phase"]!="ready": time.sleep(.2)
a=server.AGENTS[0];print(a['persona_src'],a['schedule_src'],[x[1] for x in a['schedule']])
c=server.app.test_client()
r=c.post('/api/event',json={'kind':'announce','text':'புதிய வரி','sim':600}).json;print(r['src'],[(x['id'],x['action'],x['src']) for x in r['reactions']][-3:])
for i in range(12): c.post('/api/say',json={'id':'Afraa','act':'x','place':'cafe','sim':700+i})
m=c.get('/api/memory/Afraa').json;print('long',len(m['long']),[x['kind'] for x in m['long']][-3:],'short',len(m['short']))
print(c.post('/api/converse',json={'a':'Afraa','b':'Surya','place':'cafe','sim':1}).json['src'])
