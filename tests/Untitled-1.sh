echo 'https://placekitten.com/200/305' >> myimglist.txt
echo 'https://placekitten.com/200/304' >> myimglist.txt
echo 'https://placekitten.com/200/303' >> myimglist.txt

img2dataset --url_list=myimglist.txt --output_folder=output_folder --thread_count=64 --image_size=256
