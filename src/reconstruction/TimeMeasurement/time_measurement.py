def convert(seconds):
	
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hour, minutes, seconds

"""
Usage:
st = time.process_time()
code …
et = time.process_time()
res = et - st
hours minutes, seconds  convert(res)
print(" ")
print('CPU Execution time: {} hours, {} Minutes, {} seconds' .format(int(hours), int(minutes), int(seconds)))
"""