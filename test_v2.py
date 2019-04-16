from mesa_star_class import MESA_STAR as ms


name1 = '2.5000_0.0200_0.0160'



star = ms.from_string(name1, history_path = '/vol/aibn1107/data2/schanlar/Thesis_work/singles/Case_B_urca/2p5M_carbon_removed/LOGS')


t = ['2.5000_0.0200_0.0160', 'asinglestring',
     '2.5000_0.0200_0.016s', '2.5000_0.200_0.0160']

for name in t:
    if ms.name_is_valid(name):
        print("Valid name!")
    else:
        print(f'problem with {name}')
