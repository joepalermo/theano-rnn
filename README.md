to get nltk punkt:
1) Start the Python interpreter
2) import nltk
3) nltk.donwload('punkt')

if you run into the following error message at step 3:

[nltk_data] Error loading punkt: <urlopen error [SSL:
[nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed
[nltk_data]     (_ssl.c:749)>
False

You may find this workaround useful:

(https://stackoverflow.com/questions/41348621/ssl-error-downloading-nltk-data)

1) Run the following command:
bash /Applications/Python 3.6/Install Certificates.command
