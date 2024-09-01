import datetime

import openai
import argparse
import time


def parse_args():
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument('--data_path', type=str, default='yelp_clean')
    parse_args.add_argument('--transfercode', type=int, default=0)
    parse_args.add_argument('--openaikey', type=str, default='')
    args = parse_args.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()

    openai.api_key = args.openaikey

    data_path = "./datasets/" + args.data_path + "/test." + str(args.transfercode)
    out_file = "./datasets/" + args.data_path + "/output4." + str(args.transfercode)

    if args.data_path == 'yelp_clean':  # v2prompt
        if args.transfercode == 0:
            prompt = "Rewrite the following sentence, maintain the content and change the sentiment of the sentence from negative to positive:"
        elif args.transfercode == 1:
            prompt = "Rewrite the following sentence, maintain the content and change the sentiment of the sentence from positive to negative:"
    
    if args.data_path == 'amazon_clean':  # v3prompt
        if args.transfercode == 0:
            prompt = "Rewrite the following amazon comment, maintain the content and change the sentiment of the sentence from negative to positive:"
        elif args.transfercode == 1:
            prompt = "Rewrite the following amazon comment, maintain the content and change the sentiment of the sentence from positive to negative:"

    if args.data_path == 'gyafc_em_clean' or args.data_path == 'gyafc_fr_clean':  # v2prompt
        if args.transfercode == 0:
            data_path = "./datasets/" + args.data_path + "/test.informal"
            prompt = "Change the style of the following sentence from informal to formal:"
        elif args.transfercode == 1:
            data_path = "./datasets/" + args.data_path + "/test.formal"
            prompt = "Change the style of the following sentence from formal to informal:"

    if args.data_path == 'shakespeare_clean':  # v2prompt
        if args.transfercode == 0:
            prompt = "Change the style of the following sentence from Shakespeare English to modern English:"
        elif args.transfercode == 1:
            prompt = "Change the style of the following sentence from modern English to Shakespeare English:"

    if args.data_path == 'styleptb_ARR_clean' :  # v2prompt
        if args.transfercode == 0:
            prompt = "Remove adjectives and adverbs of the following sentence:"

    if args.data_path == 'styleptb_TFU_clean' :  # v2prompt
        if args.transfercode == 0:
            prompt = "Convert the following sentence into the future tense:"


    num = 0
    # generate = []
    with open(data_path, "r") as f, open(out_file, "w", newline='') as g:
        for i in f.readlines():
            flag = True
            input = prompt + i.strip()
            while flag:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4-0125-preview",  # gpt-3.5-turbo-1106
                        messages=[{"role": "user", "content": input}],
                        temperature=0,
                        top_p=1,
                        max_tokens=100
                    )
                    flag = False
                except Exception as e:
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), f"Error occurred: {e}")
                    time.sleep(20)
                    pass

            g.write(response['choices'][0]['message']['content'] + "\n")
            # g.write(completion.strip('\n')+'\n')
            cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            num += 1
            # if num % 10 == 0:
            #     print(cur_time, num)
            print(cur_time, num)

            time.sleep(10)
