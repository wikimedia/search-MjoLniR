Vagrant.configure("2") do |config|

    config.vm.provider :virtualbox do |vb, override|
        override.vm.box = 'debian/contrib-jessie64'
        vb.customize ['modifyvm', :id, '--memory', '2048']
    end

    root_share_options = { id: 'vagrant-root' }
    root_share_options[:type] = :nfs
    root_share_options[:mount_options] = ['noatime', 'rsize=32767', 'wsize=3267', 'async']
    config.nfs.map_uid = Process.uid
    config.nfs.map_gid = Process.gid
    config.vm.synced_folder ".", "/vagrant", root_share_options

    config.vm.hostname = "MjoLniR"
    config.vm.network "private_network", type: "dhcp"
    config.vm.provision "shell", path: "bootstrap-vm.sh"
end
