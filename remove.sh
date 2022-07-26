for file in results/*/*;do
  echo $file
  if [[ "$file" == *"monitor"* ]];then
    rm $file
  fi
done