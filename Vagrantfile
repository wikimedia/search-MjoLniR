Vagrant.configure("2") do |config|

    config.vm.provider :virtualbox do |vb, override|
        override.vm.box = "trusty-cloud"
        override.vm.box_url = 'https://cloud-images.ubuntu.com/vagrant/trusty/current/trusty-server-cloudimg-amd64-vagrant-disk1.box'
        override.vm.box_download_insecure = true
        override.vm.synced_folder ".", "/vagrant", :mount_options => ["dmode=777"]
        vb.customize ['modifyvm', :id, '--memory', '2048']
    end

    config.vm.hostname = "MjoLniR"

    config.vm.provision "shell", path: "bootstrap-vm.sh"
end
