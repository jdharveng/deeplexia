import pickle
from os import path
import pandas as pd

# THIN ABOUT DIFFERENT ENPUT LISTS

##LOADING THE  LISTS
with open('Saved_Variables/sub_sentence_list.pkl', 'rb') as f:
    list_sentences = pickle.load(f)

# Check if a status has been changed before
if not path.exists("Saved_Variables/saved_status.pkl"):
    n_history = [0]
    with open('Saved_Variables/closest_emojis_list.pkl', 'rb') as f:
        list_emojis = pickle.load(f)

else:
    with open('Saved_Variables/saved_status.pkl', 'rb') as f:
        saved_status = pickle.load(f)
    n_history = saved_status["last_row_num "]
    list_emojis = saved_status["chosen_emoji"]


# Cumulated jobs done so far
N = sum(n_history)

##########################

# List that will remember the users choices
chosen_emoji = []


for i, emojis in enumerate(list_emojis[N:]):
    print("")
    print("HERE 's CURRENT SENTENCE <-> EMOJI COMBINATION")
    print("------------------------------------------------")
    print(list_sentences[N+i])
    print("")
    print(list(enumerate(emojis)))
    print("")
    print("     IF you want to keep the first emoji of the list type : f ")
    print("     IF you want to choose another emoji of the list type : o ")
    print("     IF you don't want to keep any of the emojis type: n")
    print("     IF you want to stop for now and save the changes made type: s")
    print("     If you're FINESHED and want to save to a DATAFRAME type: df")

    answer = input()

    if answer == 'f':
      chosen_emoji.append(emojis[0])

    elif answer == 'o':
      print("Enter a number between 0 and ", len(emojis)-1, "indicating which emoji you want to keep")
      answer = input()
      #if int(answer) > len(list_emojis[0]):
      #  print("Wrong number")
      #  print("Please enter a number between 0 and ",len(list_emojis[0]), "indicating which emoji you want to keep")
      #  answer = input()
      #else:
      chosen_emoji.append(emojis[int(answer)])


    elif answer == 'n':
      chosen_emoji.append("")

    elif answer == 's':
      # We'll save the number of the row we arrived at and all the chosen emojis
      saved_changes = list_emojis.copy()
      saved_changes[N:N+len(chosen_emoji)] = chosen_emoji
      updated_n = n_history.copy()
      updated_n.append(i)
      saved_status = {"last_row_num " : updated_n, "chosen_emoji": saved_changes}
      with open('Saved_Variables/saved_status.pkl', 'wb') as f:
          pickle.dump(saved_status, f)

      print("The current status of the changes has been saved ")
      break

    elif answer == 'df':
      saved_changes = list_emojis.copy()
      saved_changes[N:N+len(chosen_emoji)] = chosen_emoji
      # Build the dataframe from the 2 lists
      help_dict = {"sentences": list_sentences, "emoji_label": saved_changes }
      improved_dataset_df = pd.DataFrame(help_dict)
      with open('Saved_Variables/improved_dataset.pkl', 'wb') as f:
        pickle.dump(improved_dataset_df, f)
      break

    else:
      print("As you didn't entered a valid key the current status has been saved")
      saved_changes = list_emojis.copy()
      saved_changes[N:N+len(chosen_emoji)] = chosen_emoji
      updated_n = n_history.copy()
      updated_n.append(i)

      saved_status = {"last_row_num " : updated_n, "chosen_emoji": saved_changes}
      with open('Saved_Variables/saved_status.pkl', 'wb') as f:
          pickle.dump(saved_status, f)

      print("The current status of the changes has been saved ")
      break




