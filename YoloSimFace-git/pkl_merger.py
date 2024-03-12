import pickle
def override_frames_for_mother_child(origf, childf,motherf, startf):
    with open(origf, 'rb') as f:
        orig_dict = pickle.load(f)
    with open(childf, 'rb') as f:
        child_dict = pickle.load(f)
    with open(motherf, 'rb') as f:
        mother_dict = pickle.load(f)
    print(len(orig_dict))
    # Override frames in the original dictionary with frames from the new dictionary
    last_key = list(child_dict.keys())[-1]
    total_frames = int(last_key.split('frame')[1])
    for i in range(total_frames+1):
        if "frame"+str(i) in child_dict.keys():
            child_emo = child_dict["frame"+str(i)]
            try:
                orig_dict['frame' + str(startf + i)]['child'] = child_emo
            except:
                orig_dict['frame' + str(startf + i)] = {'child':child_emo}
            if "frame"+str(i) in mother_dict.keys():
                mother_emo = mother_dict["frame"+str(i)]
                try:
                    orig_dict['frame' + str(startf + i)]['mother'] = mother_emo
                except:
                    orig_dict['frame' + str(startf + i)] = {'mother':mother_emo}
    print(len(orig_dict))
    with open(origf, 'wb') as f:
        pickle.dump(orig_dict, f)
    with open(origf, 'rb') as f:
        orig_dict = pickle.load(f)
        for line in orig_dict:
            print(line+str(orig_dict[line]))

# Example usage:
override_frames_for_mother_child('overrideFrames/emotion_res_orig.pkl', 
                'overrideFrames/emotion_res_child.pkl',
                'overrideFrames/emotion_res_mother.pkl', 
                8303)

def flip_mother_child(origf, startf, endf):
    with open(origf, 'rb') as f:
        orig_dict = pickle.load(f)
        for i in range(startf, endf+1):
            try:
                value_dict = orig_dict['frame' + str(i)]
                print(f"Original values for frame{str(i)}:", value_dict)
                if 'mother' in value_dict and 'child' in value_dict:
                    child_emo = value_dict['child']
                    mother_emo = value_dict['mother']
                    value_dict['mother'], value_dict['child'] = child_emo, mother_emo
                elif 'mother' in value_dict:
                    value_dict['child'] = value_dict.pop('mother')
                elif 'child' in value_dict:
                    value_dict['mother'] = value_dict.pop('child')
                orig_dict['frame' + str(i)] = value_dict
                print(f"Updated values for frame{str(i)}:", value_dict)
            except KeyError:
                pass
    with open(origf, 'wb') as f:
        pickle.dump(orig_dict, f)

# Call the function to test
flip_mother_child('overrideFrames/emotion_res.pkl',6085,20497)


import pickle

def delete_certain_frames(origf, startf, endf, identity):
    with open(origf, 'rb') as f:
        orig_dict = pickle.load(f)
        for i in range(startf, endf+1):
            try:
                value_dict = orig_dict['frame' + str(i)]
                print(f"Original values for frame{str(i)}:", value_dict)
                if identity in value_dict:
                    value_dict.pop(identity)
                if len(value_dict)!=0:
                    orig_dict['frame' + str(i)] = value_dict
                print(f"Updated values for frame{str(i)}:", value_dict)
            except KeyError:
                pass
    with open(origf, 'wb') as f:
        pickle.dump(orig_dict, f)

delete_certain_frames('overrideFrames/emotion_res.pkl',6085,20497,'child')