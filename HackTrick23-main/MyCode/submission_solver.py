import numpy as np
import random
import json
import time
import json
import base64
import jwt
import pyshark
from jwcrypto import jwk
import jwt
import warnings
warnings.filterwarnings("ignore")
import gym_maze
import requests
import numpy as np
from PIL import Image
from gym_maze.envs.maze_manager import MazeManager
rids_solved = []
paths =[]
rid_time = {}
G_moves = {}
prev_acc = []
import easyocr
reader = easyocr.Reader(['en'])
    



def decode_jwt(token1):
    decoded = jwt.decode(token1, options={"verify_signature": False, "verify_aud": False, "audience": "my_audience"})
    iss1 = decoded.get("iss", "")
    aud1 = decoded.get("aud", "")
    id1 = decoded.get("id", "")
    scope1 = decoded.get("scope", "")
    name1 = decoded.get("name", "")
    email1 = decoded.get("email", "")
    rand1 = decoded.get("rand", "")
    return iss1, aud1, id1, scope1, name1, email1, rand1


def split_binary(binary_num, chunk_size):
    # Make sure the binary number is a string
    binary_str = str(binary_num)
    # Split the binary string into chunks of the given size
    binary_chunks = [binary_str[i:i + chunk_size] for i in range(0, len(binary_str), chunk_size)]

    return binary_chunks


def cipher_solver(encoded_string):
    # Add padding characters to the input string
    padding = len(encoded_string) % 4
    if padding > 0:
        encoded_string += '=' * (4 - padding)

    decoded_bytes = base64.b64decode(encoded_string)
    decoded_string = decoded_bytes.decode('utf-8')

    binary_str = ''
    number_str = ''
    for char in decoded_string:
        if char.isdigit():
            number_str += char
        elif char == ',':
            binary_str += number_str
            number_str = ''

    tuple_str = '(' + binary_str + ',' + number_str + ')'

    split_tuple = tuple_str[1:-1].split(',')

    binary_str = split_tuple[0]
    integer_val = int(split_tuple[1])
    decimal_val = int(str(integer_val), 2)
    length = len(binary_str)
    parts=int(length/decimal_val)

    binary_num = binary_str   # first input after base64 decode
    chunk_size = parts
    binary_chunks = split_binary(binary_num, chunk_size)
    output_string = ' '.join(binary_chunks)

    binary_chunks = [binary_num[i:i+parts] for i in range(0, len(binary_num), parts)]

    decimal_nums = []
    for chunk in binary_chunks:
        decimal_nums.append(int(chunk, 2))


    new_list = []

    for item in decimal_nums:
        if item >= 65 and item <= 90:
            item -= decimal_val
            if item >= 65 and item <= 90:
                new_list.append(item)
            else:
                item += decimal_val+ decimal_val + 8
                new_list.append(item)
        elif item >= 97 and item <= 122:
            item -= decimal_val
            if item >= 97 and item <= 122:
                new_list.append(item)
            else:
                item += decimal_val+ decimal_val + 8
                new_list.append(item)
    ascii_list = new_list
    char_list = []

    for ascii_num in ascii_list:
        char_list.append(chr(ascii_num))
    final=''.join(char_list)
    return final


def captcha_solver(mylist):
    my_array = np.array(mylist)
    captcha_image = Image.fromarray(my_array.astype('uint8'))
    captcha_image = np.array(captcha_image)
    captcha_image = captcha_image[ :, ::].copy() 
    # Convert the numpy array to a Pillow image
    
    # Use pytesseract to convert the image to plain text
    result = reader.readtext(captcha_image)

    # Print the plain text
    return  result[0][-2]


def pcap_solver(stri):
    
    decoded_data = base64.b64decode(stri)
    
    with open('./output.bin', 'wb') as f:
        f.write(decoded_data)

    capture = pyshark.FileCapture('./output.bin')

    packet_str = ""  # Initialize an empty string to store the packets

    for packet in capture:
        # Append the packet summary to the packet string
        packet_str += str(packet) + "\n"

    capture.close()

    pattern = "google.com:"

    found_lines = set()
    list=[]
    newlist=[None]*4


    for line in packet_str.splitlines():
        if pattern in line and line not in found_lines:
            found_lines.add(line)
            domain = line.split()[0]  # extract the first word (domain name)
            subdomain = domain.split('.')[1]  # extract the second-to-last part of the domain name
            list.append(subdomain)

    newlist[0]=list[3]
    newlist[1]=list[2]
    newlist[2]=list[0]
    newlist[3]=list[1]

    decoded_list = []

    for encoded_str in newlist:
        decoded_bytes = base64.b64decode(encoded_str)
        decoded_str = decoded_bytes.decode('utf-8')
        decoded_list.append(decoded_str)



    final_string = ''.join(decoded_list)
    return final_string

def server_solver(token1):
    
    with open("keypair.pem", "rb") as pemfile:
        key = jwk.JWK.from_pem(pemfile.read())
        public_key = key.export(private_key=False)

    jwks = {
        "e": key['e'],
        "kid": key.key_id,
        "kty": "RSA",
        "n": key['n']
    }

    header = {
        "kid": key.key_id,
        "typ": "JWT",
        "alg": "RS256",
        "jwk": jwks
    }

    fields = decode_jwt(token1)

    iss1, aud1, id1, scope1, name1, email1, rand1 = fields

    claims = {
        "iss": iss1,
        "aud": aud1,
        "id": id1,
        "scope": scope1,
        "name": name1,
        "email": email1,
        "admin": "true",
        "rand": rand1
    }
    key_pem = key.export_to_pem(private_key=True, password=None)
    signed_token = jwt.encode(payload=claims, key=key_pem, algorithm='RS256', headers=header)
    
    return signed_token.decode('utf-8')


# Loop through all possible points
for i in range(10):
    for j in range(10):
        # Create a tuple of the current point
        point = f"[{str(i)} {str(j)}]"
        # Set the value of the dictionary to ["N","S","E","W"]
        G_moves[str(point)] = ["N","S","E","W"]

def elemenate(action):
    c_point = paths[-1]
    if(action =='N'):
        rev = 'S'
        next_point =  c_point[:3]+ str(int(c_point[3])-1)+c_point[4:]
    elif(action =='S'):
        rev = 'N'
        next_point =  c_point[:3]+ str(int(c_point[3])+1)+c_point[4:]
    elif(action =='W'):
        rev = 'E'
        next_point =  c_point[:1]+ str(int(c_point[1])-1)+c_point[2:]
    elif(action =='E'):
        rev = 'W'
        next_point =  c_point[:1]+ str(int(c_point[1])+1)+c_point[2:]
    G_moves[next_point].remove(rev)


def save_action(action):
    prev_acc.append(action)


def cheak_point():
    if paths[-1] == paths[-2]:
        paths.pop()
        G_moves[paths[-1]].remove(prev_acc[-1])

### the api calls must be modified by you according to the server IP communicated with you
#### students track --> 16.170.85.45
#### working professionals track --> 13.49.133.141
server_ip = '16.170.85.45'

def select_action(state):
    actions = ['N', 'S', 'E', 'W']
    paths.append(str(state[0])) 
    if len(paths) >=2:
        cheak_point()
        available_values = G_moves[str(state[0])]
        random_action =random.choice(available_values)
        if len(available_values) == 1:
            elemenate(random_action)
        save_action(random_action)
        action_index = actions.index(random_action)
        return random_action, action_index

    else:
        available_values = actions
        random_action =random.choice(available_values)
        save_action(random_action)
        action_index = actions.index(random_action)
        return random_action, action_index


def move(agent_id, action):
    response = requests.post(f'http://{server_ip}:5000/move', json={"agentId": agent_id, "action": action})
    return response

def solve(agent_id,  riddle_type, solution):
    response = requests.post(f'http://{server_ip}:5000/solve', json={"agentId": agent_id, "riddleType": riddle_type, "solution": solution}) 
    print(response.json()) 
    return response

def get_obv_from_response(response):
    directions = response.json()['directions']
    distances = response.json()['distances']
    position = response.json()['position']
    obv = [position, distances, directions] 
    return obv

        
def submission_inference(riddle_solvers):

    response = requests.post(f'http://{server_ip}:5000/init', json={"agentId": agent_id})
    print(response)
    obv = get_obv_from_response(response)
    t = 0
    while(True):
        t+=1
        # Select an action
        state_0 = obv
        action, action_index = select_action(state_0) # Random action
        response = move(agent_id, action)
        if not response.status_code == 200:
            print(response)
            break

        obv = get_obv_from_response(response)
        print(response.json())

        if not response.json()['riddleType'] == None:
            if  response.json()['riddleType'] not in rids_solved:
                rids_solved.append(response.json()['riddleType'])
                solution = riddle_solvers[response.json()['riddleType']](response.json()['riddleQuestion'])
                response = solve(agent_id, response.json()['riddleType'], solution)


        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        if (np.array_equal(response.json()['position'], (9,9)) and t>=1000) or (np.array_equal(response.json()['position'], (9,9)) and len(rids_solved) >=3 ):
            response = requests.post(f'http://{server_ip}:5000/leave', json={"agentId": agent_id})
            break


if __name__ == "__main__":
    
    agent_id = "Nk3DzF6QpM"
    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    submission_inference(riddle_solvers)
    
